#!/usr/bin/env python3
"""Diagnose HiDe/my_method2 anchor expert selection from a saved checkpoint."""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel, CLIPVisionModelWithProjection


ROOT = Path("/mnt/lyaa/my_llava")
DEFAULT_CKPT = ROOT / "checkpoint/UCIT/LLaVA-1.5/my_method2/static/Task2_llava_lora"
DEFAULT_DATA = ROOT / "UCIT/ArxivQA/test_3000.json"
DEFAULT_IMAGE_FOLDER = ROOT / "UCIT/datasets"
DEFAULT_CLIP = ROOT / "clip-vit-large-patch14-336"
DEFAULT_LLAVA = ROOT / "llava-v1.5-7b"


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_anchors(ckpt: Path, expert_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    state = torch.load(ckpt / "non_lora_trainables.bin", map_location="cpu")

    def collect(prefix: str) -> torch.Tensor:
        values = []
        for idx in range(expert_num):
            candidates = [
                f"base_model.model.{prefix}.{idx}",
                f"model.{prefix}.{idx}",
                f"{prefix}.{idx}",
            ]
            for key in candidates:
                if key in state:
                    values.append(state[key].float().reshape(-1))
                    break
            else:
                raise KeyError(f"Could not find {prefix}.{idx} in {ckpt / 'non_lora_trainables.bin'}")
        return torch.stack(values, dim=0)

    return collect("image_anchors"), collect("text_anchors")


def build_clip_text_from_sample(sample: dict, llava_tokenizer=None, text_mode: str = "simple") -> str:
    if "conversations" in sample and sample["conversations"]:
        human = sample["conversations"][0]["value"]
    else:
        human = sample.get("text") or sample.get("question") or sample.get("prompt") or ""
    question = human.replace("<image>", "").strip()
    if text_mode == "simple" or llava_tokenizer is None:
        return question if question else "image question"

    # Match llava_arch.py's eval-time text extraction as closely as possible:
    # decode prompt tokens with image token replaced by pad, drop the first line,
    # and keep content before " ASSISTANT".
    try:
        sys.path.insert(0, str(ROOT / "LLaVA/my_method2"))
        from llava import conversation as conversation_lib
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], human)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, llava_tokenizer, return_tensors="pt")
        pad_id = llava_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = llava_tokenizer.unk_token_id
        input_ids = input_ids.clone()
        input_ids[input_ids == IMAGE_TOKEN_INDEX] = pad_id
        decoded = llava_tokenizer.batch_decode(input_ids.unsqueeze(0), skip_special_tokens=True)[0]
        hidden = "\n".join(decoded.split("\n")[1:])
        clipped = hidden.split(" ASSISTANT")[0].strip()
        return clipped if clipped else (question if question else "image question")
    except Exception:
        return question if question else "image question"


def load_samples(path: Path, num_samples: int, seed: int, mode: str) -> List[dict]:
    data = load_json(path)
    if num_samples <= 0 or num_samples >= len(data):
        return data
    if mode == "head":
        return data[:num_samples]
    rng = random.Random(seed)
    indices = rng.sample(range(len(data)), num_samples)
    return [data[i] for i in indices]


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    values = sorted(float(v) for v in values)

    def pct(q: float) -> float:
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    return {
        "mean": sum(values) / len(values),
        "median": pct(0.5),
        "p05": pct(0.05),
        "p95": pct(0.95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--data-json", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--image-folder", type=Path, default=DEFAULT_IMAGE_FOLDER)
    parser.add_argument("--clip-path", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--llava-model", type=Path, default=DEFAULT_LLAVA)
    parser.add_argument("--expert-num", type=int, default=6)
    parser.add_argument("--current-expert", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--sample-mode", choices=["random", "head"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--text-mode",
        choices=["simple", "llava_prompt"],
        default="simple",
        help="Use raw question text by default; llava_prompt more closely mimics llava_arch.py but imports llava code.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    image_anchors, text_anchors = load_anchors(args.checkpoint, args.expert_num)
    image_anchors = image_anchors.to(device)
    text_anchors = text_anchors.to(device)

    processor = CLIPImageProcessor.from_pretrained(args.clip_path)
    vision = CLIPVisionModelWithProjection.from_pretrained(args.clip_path).to(device).eval()
    text_model = CLIPTextModel.from_pretrained(args.clip_path).to(device).eval()
    clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_path, use_fast=False)
    llava_tokenizer = None
    if args.text_mode == "llava_prompt":
        llava_tokenizer = AutoTokenizer.from_pretrained(args.llava_model, use_fast=False)
        if llava_tokenizer.pad_token is None:
            llava_tokenizer.pad_token = llava_tokenizer.unk_token

    samples = load_samples(args.data_json, args.num_samples, args.seed, args.sample_mode)
    top_counts = Counter()
    image_top_counts = Counter()
    text_top_counts = Counter()
    weights_by_expert = [[] for _ in range(args.expert_num)]
    sims_by_expert = [[] for _ in range(args.expert_num)]
    current_minus_old = []
    current_weight = []
    old_weight = []
    current_rank = []

    failures = 0
    with torch.no_grad():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start : start + args.batch_size]
            images = []
            texts = []
            for sample in batch:
                try:
                    image = Image.open(args.image_folder / sample["image"]).convert("RGB")
                    text = build_clip_text_from_sample(sample, llava_tokenizer, args.text_mode)
                    images.append(image)
                    texts.append(text if text.strip() else "image question")
                except Exception:
                    failures += 1

            if not images or not texts:
                continue
            if len(images) != len(texts):
                failures += abs(len(images) - len(texts))
                continue

            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            text_inputs = clip_tokenizer(
                texts,
                padding="longest",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            image_features = vision(pixel_values=pixel_values, output_hidden_states=True).image_embeds
            text_features = text_model(**text_inputs).pooler_output

            image_sim = F.cosine_similarity(image_features[:, None, :], image_anchors[None, :, :], dim=-1)
            text_sim = F.cosine_similarity(text_features[:, None, :], text_anchors[None, :, :], dim=-1)
            sim = (image_sim + text_sim) / 2.0
            weights = F.softmax(sim / args.temperature, dim=-1)

            image_top = image_sim.argmax(dim=-1).detach().cpu().tolist()
            text_top = text_sim.argmax(dim=-1).detach().cpu().tolist()
            top = weights.argmax(dim=-1).detach().cpu().tolist()
            rank_order = weights.argsort(dim=-1, descending=True).detach().cpu().tolist()

            for row_idx, expert in enumerate(top):
                top_counts[int(expert)] += 1
                image_top_counts[int(image_top[row_idx])] += 1
                text_top_counts[int(text_top[row_idx])] += 1
                row_weights = weights[row_idx].detach().cpu().tolist()
                row_sims = sim[row_idx].detach().cpu().tolist()
                for exp_idx in range(args.expert_num):
                    weights_by_expert[exp_idx].append(row_weights[exp_idx])
                    sims_by_expert[exp_idx].append(row_sims[exp_idx])
                current_weight.append(row_weights[args.current_expert])
                old_weight.append(row_weights[0])
                current_minus_old.append(row_weights[args.current_expert] - row_weights[0])
                current_rank.append(rank_order[row_idx].index(args.current_expert) + 1)

    total = sum(top_counts.values())
    print(f"[anchor selection diagnostic]")
    print(f"checkpoint={args.checkpoint}")
    print(f"data={args.data_json} samples={total} failures={failures} device={device}")
    print(f"current_expert={args.current_expert} temperature={args.temperature} text_mode={args.text_mode}")

    print("\n[top expert by final anchor weight]")
    for exp_idx in range(args.expert_num):
        count = top_counts[exp_idx]
        pct = 100.0 * count / max(1, total)
        print(f"expert{exp_idx}: {count}/{total} ({pct:.2f}%)")

    print("\n[top expert by image/text similarity]")
    for exp_idx in range(args.expert_num):
        image_pct = 100.0 * image_top_counts[exp_idx] / max(1, total)
        text_pct = 100.0 * text_top_counts[exp_idx] / max(1, total)
        print(f"expert{exp_idx}: image_top={image_pct:.2f}% text_top={text_pct:.2f}%")

    print("\n[mean final softmax weights]")
    for exp_idx in range(args.expert_num):
        w = summarize(weights_by_expert[exp_idx])
        s = summarize(sims_by_expert[exp_idx])
        print(
            f"expert{exp_idx}: weight_mean={w['mean']:.4f} weight_median={w['median']:.4f} "
            f"weight_p05={w['p05']:.4f} weight_p95={w['p95']:.4f} sim_mean={s['mean']:.4f}"
        )

    margin = summarize(current_minus_old)
    cur_w = summarize(current_weight)
    old_w = summarize(old_weight)
    avg_rank = sum(current_rank) / max(1, len(current_rank))
    top_rate = top_counts[args.current_expert] / max(1, total)
    print("\n[current-vs-old]")
    print(
        f"current_top_rate={100.0 * top_rate:.2f}% avg_current_rank={avg_rank:.3f} "
        f"current_weight_mean={cur_w['mean']:.4f} old_weight_mean={old_w['mean']:.4f} "
        f"current_minus_old_mean={margin['mean']:.4f} "
        f"current_minus_old_p05={margin['p05']:.4f} current_minus_old_p95={margin['p95']:.4f}"
    )

    if total > 0 and top_rate >= 0.8:
        print("\n[takeaway] Anchor usually selects the task2/current expert; anchor mis-selection is unlikely to be the main cause.")
    elif total > 0 and top_rate >= 0.5:
        print("\n[takeaway] Anchor is mixed; it may contribute, but it is unlikely to be the only cause.")
    else:
        print("\n[takeaway] Anchor often does not select the task2/current expert; anchor routing is a plausible failure mode.")


if __name__ == "__main__":
    main()
