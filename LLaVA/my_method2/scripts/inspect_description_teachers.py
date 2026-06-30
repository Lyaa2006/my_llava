import argparse
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import transformers
from PIL import Image

from llava import conversation as conversation_lib
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.train.train_MOE import (
    build_multimodal_instruction_text,
    build_single_turn_prompt,
    find_all_linear_names,
    load_model_from_previous_task,
    temporary_description_snapshot,
)
from HiDe.peft import HiDeMOELoraConfig, TaskType, get_peft_model


DESCRIPTION_PROMPT = (
    "Describe the image using visual evidence: objects, attributes, shapes, colors, "
    "textures, scene context, visible text, and spatial relations."
)


def build_model_args(base_model_path: str, clip_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_name_or_path=base_model_path,
        vision_tower=clip_path,
        text_tower=clip_path,
        mm_vision_select_layer=-2,
        mm_text_select_layer=-1,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        tune_mm_mlp_adapter=False,
        freeze_backbone=False,
        pretrain_mm_mlp_adapter=None,
        cur_task=1,
        expert_num=6,
        task_embedding_dim=64,
        version="v1",
    )


def setup_tokenizer(model_args: SimpleNamespace):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.text_tower,
        model_max_length=1024,
        padding_side="right",
        use_fast=True,
    )
    return tokenizer, clip_tokenizer


def initialize_mm(model, tokenizer, clip_tokenizer, model_args, device, dtype):
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    model.get_model().initialize_text_modules(model_args=model_args, fsdp=None)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.set_tokenizer(tokenizer)
    model.set_clip_tokenizer(clip_tokenizer)
    model.set_cur_task(model_args.cur_task, model_args.expert_num)

    vision_tower = model.get_vision_tower()
    text_tower = model.get_text_tower()
    vision_tower.to(device=device, dtype=dtype)
    text_tower.to(device=device, dtype=dtype)
    if getattr(model.get_model(), "mm_projector", None) is not None:
        model.get_model().mm_projector.to(device=device, dtype=dtype)

    model.config.image_aspect_ratio = "pad"
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    return vision_tower.image_processor


def load_static_model(base_model_path: str, clip_path: str, device: torch.device, dtype: torch.dtype):
    model_args = build_model_args(base_model_path, clip_path)
    config = transformers.AutoConfig.from_pretrained(base_model_path)
    config.mm_vision_tower = model_args.vision_tower
    config.mm_text_tower = model_args.text_tower
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_text_select_layer = model_args.mm_text_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature

    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    tokenizer, clip_tokenizer = setup_tokenizer(model_args)
    image_processor = initialize_mm(model, tokenizer, clip_tokenizer, model_args, device, dtype)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, image_processor, model_args


def load_snapshot_model(
    base_model_path: str,
    clip_path: str,
    task1_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
):
    model_args = build_model_args(base_model_path, clip_path)
    config = transformers.AutoConfig.from_pretrained(base_model_path)
    config.mm_vision_tower = model_args.vision_tower
    config.mm_text_tower = model_args.text_tower
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_text_select_layer = model_args.mm_text_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature

    model = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )

    lora_config = HiDeMOELoraConfig(
        r=12,
        lora_alpha=24,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM_HiDe,
        task_embedding_dim=model_args.task_embedding_dim,
        expert_num=model_args.expert_num,
        cur_task=model_args.cur_task,
    )
    model = get_peft_model(model, lora_config)

    tokenizer, clip_tokenizer = setup_tokenizer(model_args)
    image_processor = initialize_mm(model, tokenizer, clip_tokenizer, model_args, device, dtype)
    load_model_from_previous_task(model, task1_checkpoint)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, image_processor, model_args


def load_samples(data_path: str, image_root: str, indices):
    with open(data_path, "r") as f:
        data = json.load(f)
    samples = []
    for idx in indices:
        sample = data[idx]
        image_path = os.path.join(image_root, sample["image"])
        samples.append(
            {
                "index": idx,
                "id": sample.get("id"),
                "image_path": image_path,
            }
        )
    return samples


def build_description_prompt(model_args) -> str:
    data_args = SimpleNamespace(
        is_multimodal=True,
        mm_use_im_start_end=model_args.mm_use_im_start_end,
    )
    description_text = build_multimodal_instruction_text(DESCRIPTION_PROMPT, data_args)
    return build_single_turn_prompt(description_text)


def generate_description(model, tokenizer, image_processor, model_args, sample, max_new_tokens: int):
    prompt = build_description_prompt(model_args)
    image = Image.open(sample["image_path"]).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(device=model.device, dtype=model.dtype) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(device=model.device, dtype=model.dtype)

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    conv = conversation_lib.default_conversation.copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    output_text = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)].strip()
    return output_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--task1-checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--indices", default="0,1,2")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    samples = load_samples(args.data_path, args.image_root, indices)

    runs = [
        ("static_backbone", load_static_model, nullcontext()),
        ("snapshot_task1", load_snapshot_model, None),
    ]

    for mode_name, loader_fn, _ in runs:
        print(f"\n===== {mode_name} =====", flush=True)
        if mode_name == "static_backbone":
            model, tokenizer, image_processor, model_args = loader_fn(
                args.base_model, args.clip_path, device, dtype
            )
            ctx = temporary_description_snapshot(model, None, 1)
        else:
            model, tokenizer, image_processor, model_args = loader_fn(
                args.base_model, args.clip_path, args.task1_checkpoint, device, dtype
            )
            adapter_name = getattr(model, "prev_task_adapter_name", None)
            ctx = temporary_description_snapshot(model, adapter_name, 0)

        with ctx:
            for sample in samples:
                output_text = generate_description(
                    model, tokenizer, image_processor, model_args, sample, args.max_new_tokens
                )
                print(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "index": sample["index"],
                            "id": sample["id"],
                            "image": sample["image_path"],
                            "description": output_text,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
