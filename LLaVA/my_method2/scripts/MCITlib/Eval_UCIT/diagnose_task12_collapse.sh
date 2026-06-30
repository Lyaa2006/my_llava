#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-$HARD_PATH/checkpoint/UCIT/LLaVA-1.5/my_method2/static_layerwise_full/Task2_llava_lora_task2_layerwise_static_full_20260629_080706}"
MODEL_BASE="${MODEL_BASE:-$HARD_PATH/llava-v1.5-7b}"
TEXT_TOWER="${TEXT_TOWER:-$HARD_PATH/clip-vit-large-patch14-336}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$HARD_PATH/UCIT/datasets}"
OUT_DIR="${OUT_DIR:-$HARD_PATH/results/UCIT/collapse_probe/my_method2_task2_layerwise_static_full_20260629_080706}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_TASK="${NUM_TASK:-6}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUT_DIR"

make_first_n() {
    local src="$1"
    local dst="$2"
    "$PYTHON_BIN" - "$src" "$dst" "$NUM_SAMPLES" <<'PY'
import json
import os
import sys

src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data[:n], f, ensure_ascii=False, indent=2)
print(f"Wrote {min(n, len(data))} examples to {dst}")
PY
}

check_environment() {
    "$PYTHON_BIN" - "$PROJECT_ROOT" <<'PY'
import os
import sys
import llava

project_root = os.path.realpath(sys.argv[1])
llava_file = os.path.realpath(llava.__file__)
print(f"llava loaded from: {llava_file}")
if not llava_file.startswith(project_root + os.sep):
    raise SystemExit(f"Expected llava from {project_root}, got {llava_file}")
PY
}

run_probe() {
    local label="$1"
    local question_file="$2"
    local answers_file="$OUT_DIR/${label}.answers.jsonl"
    local time_file="$OUT_DIR/${label}.time.txt"

    echo "========== ${label}: generate first ${NUM_SAMPLES} samples =========="
    echo "answers_file=$answers_file"
    /usr/bin/time -f "elapsed_sec=%e maxrss_kb=%M" -o "$time_file" \
        "$PYTHON_BIN" -m llava.eval.CoIN.model_others \
            --model-path "$CHECKPOINT" \
            --model-base "$MODEL_BASE" \
            --question-file "$question_file" \
            --image-folder "$IMAGE_FOLDER" \
            --text-tower "$TEXT_TOWER" \
            --num-task "$NUM_TASK" \
            --answers-file "$answers_file" \
            --num-chunks 1 \
            --chunk-idx 0 \
            --temperature 0 \
            --conv-mode vicuna_v1

    echo "========== ${label}: analyze output length/repetition =========="
    "$PYTHON_BIN" - "$answers_file" "$label" "$MODEL_BASE" "$time_file" <<'PY'
import collections
import json
import os
import re
import sys

answers_file, label, model_base, time_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

rows = []
with open(answers_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
except Exception as exc:
    print(f"[warn] tokenizer load failed; using whitespace count fallback: {exc}")

def token_count(text):
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    return len(re.findall(r"\S+", text))

def repetition_stats(text):
    units = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(units) < 4:
        return 0.0, "", 0
    grams = [tuple(units[i:i + 4]) for i in range(len(units) - 3)]
    counts = collections.Counter(grams)
    gram, freq = counts.most_common(1)[0]
    duplicate_ratio = 1.0 - len(counts) / max(1, len(grams))
    return duplicate_ratio, " ".join(gram), freq

summary = []
for i, row in enumerate(rows, 1):
    text = row.get("text", "")
    n_tokens = token_count(text)
    dup_ratio, top_4gram, top_4gram_count = repetition_stats(text)
    suspicious = n_tokens >= 230 or dup_ratio >= 0.35 or top_4gram_count >= 6
    summary.append({
        "idx": i,
        "question_id": row.get("question_id"),
        "token_count": n_tokens,
        "char_count": len(text),
        "duplicate_4gram_ratio": round(dup_ratio, 4),
        "top_4gram": top_4gram,
        "top_4gram_count": top_4gram_count,
        "suspicious": suspicious,
        "text_preview": text[:240].replace("\n", "\\n"),
    })

summary_file = os.path.join(os.path.dirname(answers_file), f"{label}.summary.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

elapsed = ""
if os.path.exists(time_file):
    with open(time_file, "r", encoding="utf-8") as f:
        elapsed = f.read().strip()

print(f"label={label}")
print(f"answers={answers_file}")
print(f"summary={summary_file}")
if elapsed:
    print(elapsed)
if summary:
    counts = [item["token_count"] for item in summary]
    suspicious = sum(1 for item in summary if item["suspicious"])
    print(
        f"samples={len(summary)} min_tokens={min(counts)} "
        f"max_tokens={max(counts)} avg_tokens={sum(counts)/len(counts):.1f} "
        f"suspicious={suspicious}"
    )

for item in summary:
    flag = "SUSPECT" if item["suspicious"] else "ok"
    print(
        f"[{flag}] #{item['idx']} qid={item['question_id']} "
        f"tokens={item['token_count']} chars={item['char_count']} "
        f"dup4={item['duplicate_4gram_ratio']} top4_count={item['top_4gram_count']} "
        f"preview={item['text_preview']}"
    )
PY
}

cd "$PROJECT_ROOT"

IMAGENET_FIRST="$OUT_DIR/ImageNet-R.first${NUM_SAMPLES}.json"
ARXIVQA_FIRST="$OUT_DIR/ArxivQA.first${NUM_SAMPLES}.json"

make_first_n "$HARD_PATH/UCIT/ImageNet-R/test_3000.json" "$IMAGENET_FIRST"
make_first_n "$HARD_PATH/UCIT/ArxivQA/test_3000.json" "$ARXIVQA_FIRST"

echo "project_root=$PROJECT_ROOT"
echo "hard_path=$HARD_PATH"
echo "checkpoint=$CHECKPOINT"
echo "model_base=$MODEL_BASE"
echo "text_tower=$TEXT_TOWER"
echo "image_folder=$IMAGE_FOLDER"
echo "num_task=$NUM_TASK"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "out_dir=$OUT_DIR"
echo "model_others max_new_tokens=256 (hard-coded in my_method2)"

check_environment

run_probe "task1_ImageNet-R" "$IMAGENET_FIRST"
run_probe "task2_ArxivQA" "$ARXIVQA_FIRST"

echo "Done. Inspect outputs under: $OUT_DIR"
