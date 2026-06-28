# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import hashlib
import json, deepspeed
import logging
import pathlib, random
from typing import Dict, Optional, Sequence, List

import torch
import sys
import transformers
import subprocess

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from HiDe.peft import TaskType, get_peft_model, HiDeMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS=None

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def build_cache_key(split: str, idx: int) -> str:
    return f"{split}_{idx}"


def get_description_cache_path(cache_dir: str, cache_key: str) -> str:
    return os.path.join(cache_dir, f"{cache_key}.pt")


def pad_description_sequences(sequences: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
    max_len = max(seq.shape[0] for seq in sequences)
    hidden_size = sequences[0].shape[-1]
    device = sequences[0].device
    dtype = sequences[0].dtype
    padded = torch.zeros((len(sequences), max_len, hidden_size), dtype=dtype, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool, device=device)
    for idx, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[idx, :seq_len] = seq
        mask[idx, :seq_len] = True
    return {"states": padded, "mask": mask}


def move_batch_to_device(batch, device):
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, list):
            moved_batch[key] = [item.to(device) if isinstance(item, torch.Tensor) else item for item in value]
        else:
            moved_batch[key] = value
    return moved_batch


def move_images_to_vision_tower(batch, model):
    images = batch.get("images")
    if images is None:
        return batch
    vision_tower = model.get_vision_tower()
    target_device = vision_tower.device
    target_dtype = vision_tower.dtype
    if isinstance(images, torch.Tensor):
        batch["images"] = images.to(device=target_device, dtype=target_dtype)
    elif isinstance(images, list):
        batch["images"] = [
            image.to(device=target_device, dtype=target_dtype) if isinstance(image, torch.Tensor) else image
            for image in images
        ]
    return batch


def move_model_to_training_device(model, training_args):
    device = training_args.device
    dtype = None
    if training_args.bf16:
        dtype = torch.bfloat16
    elif training_args.fp16:
        dtype = torch.float16

    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)

    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if dtype is None:
                vision_tower.to(device=device)
            else:
                vision_tower.to(device=device, dtype=dtype)

    if hasattr(model, "get_text_tower"):
        text_tower = model.get_text_tower()
        if text_tower is not None:
            if dtype is None:
                text_tower.to(device=device)
            else:
                text_tower.to(device=device, dtype=dtype)

    if hasattr(model, "get_model") and getattr(model.get_model(), "mm_projector", None) is not None:
        mm_projector = model.get_model().mm_projector
        if dtype is None:
            mm_projector.to(device=device)
        else:
            mm_projector.to(device=device, dtype=dtype)


@contextmanager
def temporary_description_snapshot(model, adapter_name: Optional[str], cur_task: int):
    wrapped = getattr(model, "module", model)
    prev_training = model.training
    prev_adapter = getattr(wrapped, "active_adapter", None)
    prev_task = getattr(wrapped, "cur_task", None)

    try:
        model.eval()
        if adapter_name is not None and hasattr(wrapped, "set_adapter"):
            wrapped.set_adapter(adapter_name)
        if hasattr(wrapped, "cur_task"):
            wrapped.cur_task = int(cur_task)
        yield
    finally:
        if prev_task is not None and hasattr(wrapped, "cur_task"):
            try:
                wrapped.cur_task = prev_task
            except Exception:
                pass
        if prev_adapter is not None and hasattr(wrapped, "set_adapter"):
            try:
                wrapped.set_adapter(prev_adapter)
            except Exception:
                pass
        model.train(prev_training)


def extract_description_cache_snapshot(model, tokenizer, data_args, training_args):
    if data_args.description_cache_dir is None:
        raise ValueError("`description_cache_dir` is required when extracting description cache.")
    os.makedirs(data_args.description_cache_dir, exist_ok=True)

    prev_use_description_data = getattr(data_args, "use_description_data", False)
    prev_load_description_cache = getattr(data_args, "load_description_cache", True)
    data_args.use_description_data = True
    data_args.load_description_cache = False
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    from torch.utils.data import DataLoader
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

    wrapped = getattr(model, "module", model)
    snapshot_task = int(getattr(wrapped, "cur_task", 0)) - 1
    if snapshot_task < 0:
        raise ValueError("Description cache extraction requires at least one learned historical task snapshot.")
    snapshot_adapter = getattr(wrapped, "prev_task_adapter_name", None)
    if snapshot_adapter is not None and hasattr(wrapped, "peft_config"):
        if snapshot_adapter not in getattr(wrapped, "peft_config", {}):
            snapshot_adapter = None

    use_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if use_dist else 0
    world_size = torch.distributed.get_world_size() if use_dist else 1

    cached = 0
    with torch.no_grad(), temporary_description_snapshot(model, snapshot_adapter, snapshot_task):
        for batch in loader:
            if "description_input_ids" not in batch:
                continue
            cache_keys = batch.get("description_cache_keys")
            if not cache_keys:
                continue
            cache_key = cache_keys[0]
            cache_indices = batch.get("description_cache_indices")
            cache_index = cache_indices[0] if cache_indices else None
            shard_key = (
                int(cache_index)
                if cache_index is not None
                else int(hashlib.md5(cache_key.encode("utf-8")).hexdigest(), 16)
            )
            if world_size > 1 and (shard_key % world_size) != rank:
                continue
            cache_path = get_description_cache_path(data_args.description_cache_dir, cache_key)
            if os.path.exists(cache_path):
                cached += 1
                continue

            batch = move_batch_to_device(batch, training_args.device)
            batch = move_images_to_vision_tower(batch, model)
            outputs = model(
                input_ids=batch["description_input_ids"],
                attention_mask=batch["description_attention_mask"],
                images=batch.get("images"),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[training_args.description_hidden_layer]
            lengths = batch["description_attention_mask"].long().sum(dim=1).tolist()
            cur_len = lengths[0]
            start_idx = max(0, cur_len - training_args.description_max_tokens)
            description_seq = hidden_states[0, start_idx:cur_len].detach().cpu()
            torch.save(description_seq, cache_path)
            cached += 1

    if use_dist:
        torch.distributed.barrier()

    if rank == 0:
        cache_entries = [
            name
            for name in os.listdir(data_args.description_cache_dir)
            if name.endswith(".pt")
        ]
        meta = {
            "description_prompt": data_args.description_prompt,
            "description_hidden_layer": int(training_args.description_hidden_layer),
            "description_max_tokens": int(training_args.description_max_tokens),
            "teacher": "historical_snapshot_fuse",
            "snapshot_cur_task": int(snapshot_task),
            "snapshot_adapter": snapshot_adapter or getattr(wrapped, "active_adapter", None),
            "entries": int(len(cache_entries)),
            "total": int(len(train_dataset)),
            "world_size": int(world_size),
        }
        with open(os.path.join(data_args.description_cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    data_args.use_description_data = prev_use_description_data
    data_args.load_description_cache = prev_load_description_cache


def maybe_sync_description_cache_settings(data_args, training_args, model_args=None):
    if not training_args.enable_description_cl or data_args.description_cache_dir is None:
        return

    meta_path = os.path.join(data_args.description_cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return

    with open(meta_path, "r") as f:
        cache_meta = json.load(f)

    cached_teacher = cache_meta.get("teacher")
    expected_teacher = "historical_snapshot_fuse"
    if cached_teacher != expected_teacher:
        raise ValueError(
            f"Description cache at {meta_path} was built with teacher={cached_teacher!r}, "
            f"but {expected_teacher!r} is required. Please regenerate the cache."
        )

    expected_snapshot_task = None
    if model_args is not None and getattr(model_args, "cur_task", None) is not None:
        expected_snapshot_task = int(model_args.cur_task) - 1
    if expected_snapshot_task is not None and expected_snapshot_task >= 0:
        cached_snapshot_task = cache_meta.get("snapshot_cur_task")
        if cached_snapshot_task != expected_snapshot_task:
            raise ValueError(
                f"Description cache at {meta_path} targets snapshot_cur_task={cached_snapshot_task!r}, "
                f"but current run expects {expected_snapshot_task}. Please regenerate or switch cache_dir."
            )

    cached_max_tokens = cache_meta.get("description_max_tokens")
    if cached_max_tokens is not None and cached_max_tokens != training_args.description_max_tokens:
        rank0_print(
            f"Overriding description_max_tokens from {training_args.description_max_tokens} "
            f"to cached value {cached_max_tokens} based on {meta_path}."
        )
        training_args.description_max_tokens = cached_max_tokens

    cached_hidden_layer = cache_meta.get("description_hidden_layer")
    if cached_hidden_layer is not None and cached_hidden_layer != training_args.description_hidden_layer:
        rank0_print(
            f"Overriding description_hidden_layer from {training_args.description_hidden_layer} "
            f"to cached value {cached_hidden_layer} based on {meta_path}."
        )
        training_args.description_hidden_layer = cached_hidden_layer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    text_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_text_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    cur_task: Optional[int] = field(default=0)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    memory_data_path: str = field(default=None,
                           metadata={"help": "Path to the memory data."})
    description_prompt: str = field(
        default=(
            "Describe the image using visual evidence: objects, attributes, shapes, colors, "
            "textures, scene context, visible text, and spatial relations."
        )
    )
    description_cache_dir: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    use_description_data: bool = False
    load_description_cache: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    enable_description_cl: bool = field(default=False)
    extract_description_cache_only: bool = field(default=False)
    description_hidden_layer: int = field(default=-2)
    description_max_tokens: int = field(default=32)
    description_align_weight: float = field(default=1.0)
    description_utility_weight: float = field(default=1.0)
    standard_ce_weight: float = field(default=1.0)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def build_single_turn_prompt(message: str) -> str:
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_multimodal_instruction_text(
    text: str,
    data_args: DataArguments,
) -> str:
    source = [[{"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{text}".strip()}]]
    source = preprocess_multimodal(source, data_args)
    return source[0][0]["value"]


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_train_data_dict = json.load(open(data_path, "r"))
        for idx, sample in enumerate(list_train_data_dict):
            if isinstance(sample, dict):
                sample["_description_cache_key"] = build_cache_key("train", idx)
        list_data_dict = list_train_data_dict

        if data_args.memory_data_path is not None:
            list_memory_data_dict = json.load(open(data_args.memory_data_path, "r"))
            for idx, sample in enumerate(list_memory_data_dict):
                if isinstance(sample, dict):
                    sample["_description_cache_key"] = build_cache_key("memory", idx)

            list_data_dict = list_data_dict + list_memory_data_dict
            
            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            if getattr(self.data_args, "use_description_data", False):
                description_text = build_multimodal_instruction_text(
                    self.data_args.description_prompt,
                    self.data_args,
                )
                description_prompt = build_single_turn_prompt(description_text)
                data_dict["description_input_ids"] = tokenizer_image_token(
                    description_prompt,
                    self.tokenizer,
                    return_tensors='pt',
                )
                cache_key = self.list_data_dict[i].get("_description_cache_key", build_cache_key("train", int(i)))
                data_dict["description_cache_key"] = cache_key
                data_dict["description_cache_index"] = int(i)
                if getattr(self.data_args, "load_description_cache", True) and self.data_args.description_cache_dir is not None:
                    cache_path = get_description_cache_path(self.data_args.description_cache_dir, cache_key)
                    if not os.path.exists(cache_path):
                        raise FileNotFoundError(f"Missing description cache entry: {cache_path}")
                    data_dict["reference_description_states"] = torch.load(cache_path, map_location="cpu")
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        max_len = int(getattr(self.tokenizer, "model_max_length", 2048) or 2048)
        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

        input_ids_list = []
        labels_list = []
        for instance in instances:
            cur_input_ids = instance["input_ids"]
            cur_labels = instance["labels"]
            if cur_input_ids.shape[0] > max_len:
                target_positions = torch.nonzero(cur_labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
                if target_positions.numel() == 0 or target_positions[-1].item() < max_len:
                    cur_input_ids = cur_input_ids[:max_len]
                    cur_labels = cur_labels[:max_len]
                else:
                    prefix_len = max(0, min(64, max_len - int(target_positions.numel())))
                    suffix_len = max_len - prefix_len
                    cur_input_ids = torch.cat((cur_input_ids[:prefix_len], cur_input_ids[-suffix_len:]), dim=0)
                    cur_labels = torch.cat((cur_labels[:prefix_len], cur_labels[-suffix_len:]), dim=0)
            input_ids_list.append(cur_input_ids)
            labels_list.append(cur_labels)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'description_input_ids' in instances[0]:
            description_input_ids = [instance["description_input_ids"] for instance in instances]
            description_input_ids = torch.nn.utils.rnn.pad_sequence(
                description_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            description_input_ids = description_input_ids[:, :self.tokenizer.model_max_length]
            batch["description_input_ids"] = description_input_ids
            batch["description_attention_mask"] = description_input_ids.ne(self.tokenizer.pad_token_id)
            if "description_cache_key" in instances[0]:
                batch["description_cache_keys"] = [instance.get("description_cache_key") for instance in instances]
            if "description_cache_index" in instances[0]:
                batch["description_cache_indices"] = [instance.get("description_cache_index") for instance in instances]
            if "reference_description_states" in instances[0]:
                reference_states = [instance["reference_description_states"] for instance in instances]
                padded_reference = pad_description_sequences(reference_states)
                batch["reference_description_states"] = padded_reference["states"]
                batch["reference_description_mask"] = padded_reference["mask"]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def load_model_from_previous_task(model, previous_task_model_path):
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    # if model.lm_head.weight.shape[0] != token_num:
    #     model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #     model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(previous_task_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(previous_task_model_path, 'non_lora_trainables.bin'), map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(previous_task_model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}

    model.base_model.model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    filename = os.path.join(previous_task_model_path, WEIGHTS_NAME)
    adapters_weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    load_result = set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
    prev_adapter_name = "prev_task"
    try:
        if hasattr(model, "peft_config") and prev_adapter_name not in getattr(model, "peft_config", {}):
            if hasattr(model, "add_adapter"):
                model.add_adapter(prev_adapter_name, model.peft_config["default"])
        if hasattr(model, "peft_config") and prev_adapter_name in getattr(model, "peft_config", {}):
            set_peft_model_state_dict(model, adapters_weights, adapter_name=prev_adapter_name)
            setattr(model, "prev_task_adapter_name", prev_adapter_name)
            if hasattr(model, "set_adapter"):
                model.set_adapter("default")
    except Exception as e:
        print(f"WARNING: failed to create prev_task adapter snapshot: {e}")
    print('Model is loaded...')

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.use_description_data = training_args.enable_description_cl or training_args.extract_description_cache_only
    if training_args.enable_description_cl and training_args.gradient_checkpointing:
        rank0_print(
            "Disabling gradient checkpointing because description continual-learning "
            "uses multiple gradient-carrying forwards per step."
        )
        training_args.gradient_checkpointing = False
    if training_args.enable_description_cl and data_args.description_cache_dir is None:
        raise ValueError("`enable_description_cl=True` requires `description_cache_dir`.")
    if training_args.extract_description_cache_only and data_args.description_cache_dir is None:
        raise ValueError("`extract_description_cache_only=True` requires `description_cache_dir`.")
    maybe_sync_description_cache_settings(data_args, training_args, model_args)
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            config.mm_vision_tower = model_args.vision_tower
            config.mm_text_tower = getattr(model_args, "text_tower", None) or model_args.vision_tower
            config.mm_vision_select_layer = model_args.mm_vision_select_layer
            config.mm_vision_select_feature = model_args.mm_vision_select_feature
            config.mm_text_select_layer = model_args.mm_text_select_layer
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
            config.mm_vision_tower = model_args.vision_tower
            config.mm_text_tower = getattr(model_args, "text_tower", None) or model_args.vision_tower
            config.mm_vision_select_layer = model_args.mm_vision_select_layer
            config.mm_vision_select_feature = model_args.mm_vision_select_feature
            config.mm_text_select_layer = model_args.mm_text_select_layer
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.training = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        kwargs = { 
                "task_embedding_dim": model_args.task_embedding_dim,
                "expert_num": model_args.expert_num,
                "cur_task": model_args.cur_task,
            }
        lora_config = HiDeMOELoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM_HiDe,
            **kwargs
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.get_model().initialize_text_modules(
                model_args=model_args,
                fsdp=training_args.fsdp
            )
        text_tower = model.get_text_tower()
        text_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.text_tower,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    model.set_clip_tokenizer(clip_tokenizer)
    model.set_tokenizer(tokenizer)
    model.set_cur_task(model_args.cur_task, model_args.expert_num)

    if model_args.previous_task_model_path is not None:
        # load model from previous task
        load_model_from_previous_task(model, model_args.previous_task_model_path)
    elif training_args.enable_description_cl or training_args.extract_description_cache_only:
        raise ValueError("`previous_task_model_path` is required for offline historical snapshot description cache.")

    if training_args.extract_description_cache_only:
        move_model_to_training_device(model, training_args)
        extract_description_cache_snapshot(model, tokenizer, data_args, training_args)
        return

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        model.set_boundary_for_save()
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        mm_adapter_keys = ['mm_projector', 'vision_resampler']
        if getattr(training_args, "use_im_start_end", False):
            mm_adapter_keys.extend(['embed_tokens', 'embed_in'])
        mm_adapter_state_dict = get_mm_adapter_state_maybe_zero_3(
            model.named_parameters(), mm_adapter_keys
        )
        non_lora_state_dict.update(mm_adapter_state_dict)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            if mm_adapter_state_dict:
                torch.save(mm_adapter_state_dict, os.path.join(training_args.output_dir, 'mm_projector.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

    remove_dir = training_args.output_dir
    subprocess.run(f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +", shell=True)

if __name__ == "__main__":
    train()
