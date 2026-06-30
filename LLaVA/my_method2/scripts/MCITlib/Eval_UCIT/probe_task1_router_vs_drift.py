import argparse
import collections
import json
import os
import re
from types import MethodType

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.CoIN.coin_utils import get_model_name_from_path
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


PROJ_NAMES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=str)
    parser.add_argument("--model-base", required=True, type=str)
    parser.add_argument("--text-tower", required=True, type=str)
    parser.add_argument("--question-file", required=True, type=str)
    parser.add_argument("--image-folder", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-task", type=int, default=6)
    parser.add_argument("--task1-expert-idx", type=int, default=0)
    parser.add_argument("--task2-expert-idx", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--modes",
        type=str,
        default="auto,force_task1,force_task2",
        help="Comma-separated list from {auto,force_task1,force_task2,force_uniform}",
    )
    return parser.parse_args()


def get_last_layer_proj_modules(model):
    layer = model.model.layers[-1]
    modules = {}
    for name in PROJ_NAMES:
        if name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            modules[name] = getattr(layer.self_attn, name)
        else:
            modules[name] = getattr(layer.mlp, name)
    return modules


def read_last_layer_weights(model):
    modules = get_last_layer_proj_modules(model)
    weights = {}
    for name, module in modules.items():
        raw = getattr(module, "expert_weight", None)
        if raw is None:
            weights[name] = None
        else:
            weights[name] = [float(x) for x in raw]
    return weights


def set_last_layer_weights(model, values):
    modules = get_last_layer_proj_modules(model)
    copied = [float(x) for x in values]
    for module in modules.values():
        module.expert_weight = copied.copy()


def make_forced_weights(mode, num_task, task1_idx, task2_idx):
    if mode == "auto":
        return None
    if mode == "force_uniform":
        return [1.0 / num_task for _ in range(num_task)]
    forced = [0.0 for _ in range(num_task)]
    if mode == "force_task1":
        forced[task1_idx] = 1.0
        return forced
    if mode == "force_task2":
        forced[task2_idx] = 1.0
        return forced
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_weights(weights):
    if not weights:
        return {}
    first = next((v for v in weights.values() if v is not None), None)
    if first is None:
        return {}
    top_idx = max(range(len(first)), key=lambda i: first[i])
    return {
        "top_expert": int(top_idx),
        "top_weight": float(first[top_idx]),
        "weights": [float(x) for x in first],
    }


def repetition_stats(text):
    units = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(units) < 4:
        return 0.0, "", 0
    grams = [tuple(units[i:i + 4]) for i in range(len(units) - 3)]
    counts = collections.Counter(grams)
    gram, freq = counts.most_common(1)[0]
    dup_ratio = 1.0 - len(counts) / max(1, len(grams))
    return dup_ratio, " ".join(gram), freq


class RouteProbe:
    def __init__(self, model, num_task, task1_idx, task2_idx):
        self.model = model
        self.num_task = num_task
        self.task1_idx = task1_idx
        self.task2_idx = task2_idx
        self.mode = "auto"
        self.capture = {}
        self._orig_prepare = model.prepare_inputs_labels_for_multimodal
        model.prepare_inputs_labels_for_multimodal = MethodType(self._wrapped_prepare, model)

    def set_mode(self, mode):
        self.mode = mode
        self.capture = {}

    def _wrapped_prepare(self, model_self, input_ids, position_ids, attention_mask, past_key_values, labels, images):
        result = self._orig_prepare(input_ids, position_ids, attention_mask, past_key_values, labels, images)
        auto_weights = read_last_layer_weights(self.model)
        forced_values = make_forced_weights(self.mode, self.num_task, self.task1_idx, self.task2_idx)
        if forced_values is not None:
            set_last_layer_weights(self.model, forced_values)
        applied_weights = read_last_layer_weights(self.model)
        self.capture = {
            "mode": self.mode,
            "auto": summarize_weights(auto_weights),
            "applied": summarize_weights(applied_weights),
        }
        return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        num_task=args.num_task,
        text_tower=args.text_tower,
    )
    model.eval()

    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = json.load(f)[: args.num_samples]

    probe = RouteProbe(model, args.num_task, args.task1_expert_idx, args.task2_expert_idx)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    results = []
    results_path = os.path.join(args.output_dir, "task1_router_vs_drift_results.jsonl")
    summary_path = os.path.join(args.output_dir, "task1_router_vs_drift_summary.json")

    with open(results_path, "w", encoding="utf-8") as out_f:
        for idx, line in enumerate(tqdm(questions, desc="samples"), start=1):
            image_file = line["image"]
            prompt_text = line["text"]
            question_id = line["question_id"]

            for mode in modes:
                probe.set_mode(mode)

                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).cuda()

                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=args.temperature > 0,
                        temperature=args.temperature,
                        num_beams=1,
                        max_new_tokens=256,
                        stopping_criteria=[stopping_criteria],
                        use_cache=True,
                    )

                input_token_len = input_ids.shape[1]
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()

                dup_ratio, top_4gram, top_4gram_count = repetition_stats(outputs)
                result = {
                    "sample_idx": idx,
                    "question_id": question_id,
                    "image": image_file,
                    "mode": mode,
                    "prompt": prompt_text,
                    "text": outputs,
                    "token_count": len(tokenizer(outputs, add_special_tokens=False).input_ids),
                    "char_count": len(outputs),
                    "duplicate_4gram_ratio": round(float(dup_ratio), 4),
                    "top_4gram": top_4gram,
                    "top_4gram_count": int(top_4gram_count),
                    "route_capture": probe.capture,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                }
                results.append(result)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

    grouped = collections.defaultdict(list)
    for item in results:
        grouped[item["mode"]].append(item)

    summary = {}
    for mode, items in grouped.items():
        counts = [x["token_count"] for x in items]
        summary[mode] = {
            "samples": len(items),
            "avg_tokens": sum(counts) / len(counts),
            "min_tokens": min(counts),
            "max_tokens": max(counts),
            "num_repetitive": sum(1 for x in items if x["duplicate_4gram_ratio"] >= 0.35 or x["top_4gram_count"] >= 6),
            "top_expert_histogram": dict(collections.Counter(
                x["route_capture"].get("applied", {}).get("top_expert")
                for x in items
            )),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved detailed results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
