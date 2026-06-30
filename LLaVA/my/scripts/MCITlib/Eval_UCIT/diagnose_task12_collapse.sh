#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT="${CHECKPOINT:-$HARD_PATH/checkpoint/UCIT/LLaVA-1.5/my_method2/static_layerwise_full/Task2_llava_lora_task2_layerwise_static_full_20260629_080706}"
MODEL_BASE="${MODEL_BASE:-$HARD_PATH/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$HARD_PATH/UCIT/datasets}"
OUT_DIR="${OUT_DIR:-$HARD_PATH/results/UCIT/collapse_probe/task2_layerwise_static_full_20260629_080706}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
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

run_probe() {
    local label="$1"
    local question_file="$2"
    local answers_file="$OUT_DIR/${label}.answers.jsonl"
    local time_file="$OUT_DIR/${label}.time.txt"

    echo "========== ${label}: generate first ${NUM_SAMPLES} samples =========="
    /usr/bin/time -f "elapsed_sec=%e maxrss_kb=%M" -o "$time_file" \
        "$PYTHON_BIN" -m llava.eval.CoIN.model_others \
            --model-path "$CHECKPOINT" \
            --model-base "$MODEL_BASE" \
            --question-file "$question_file" \
            --image-folder "$IMAGE_FOLDER" \
            --answers-file "$answers_file" \
            --num-chunks 1 \
            --chunk-idx 0 \
            --temperature 0 \
            --conv-mode vicuna_v1 \
            --max_new_tokens "$MAX_NEW_TOKENS"

    echo "========== ${label}: analyze output length/repetition =========="
    "$PYTHON_BIN" - "$answers_file" "$label" "$MAX_NEW_TOKENS" "$MODEL_BASE" "$time_file" <<'PY'
import collections
import json
import os
import re
import sys

answers_file, label, max_new_tokens, model_base, time_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

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
    print(f"[warn] tokenizer load failed, falling back to regex token count: {exc}")

def count_tokens(text):
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    return len(re.findall(r"\S+", text))

def repetition_stats(text):
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(words) < 4:
        return 0.0, "", 0
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    counts = collections.Counter(grams)
    gram, freq = counts.most_common(1)[0]
    duplicate_ratio = 1.0 - (len(counts) / max(1, len(grams)))
    return duplicate_ratio, " ".join(gram), freq

summary = []
for i, row in enumerate(rows, 1):
    text = row.get("text", "")
    token_count = count_tokens(text)
    duplicate_ratio, top_4gram, top_4gram_count = repetition_stats(text)
    suspicious = token_count >= max_new_tokens - 5 or duplicate_ratio >= 0.35 or top_4gram_count >= 6
    item = {
        "idx": i,
        "question_id": row.get("question_id"),
        "token_count": token_count,
        "char_count": len(text),
        "duplicate_4gram_ratio": round(duplicate_ratio, 4),
        "top_4gram": top_4gram,
        "top_4gram_count": top_4gram_count,
        "suspicious": suspicious,
        "text_preview": text[:240].replace("\n", "\\n"),
    }
    summary.append(item)

out_json = os.path.join(os.path.dirname(answers_file), f"{label}.summary.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

elapsed = ""
if os.path.exists(time_file):
    elapsed = open(time_file, "r", encoding="utf-8").read().strip()

print(f"label={label}")
print(f"answers={answers_file}")
print(f"summary={out_json}")
if elapsed:
    print(elapsed)
if summary:
    token_counts = [x["token_count"] for x in summary]
    suspicious = sum(1 for x in summary if x["suspicious"])
    print(f"samples={len(summary)} min_tokens={min(token_counts)} max_tokens={max(token_counts)} avg_tokens={sum(token_counts)/len(token_counts):.1f} suspicious={suspicious}")
for x in summary:
    flag = "SUSPECT" if x["suspicious"] else "ok"
    print(
        f"[{flag}] #{x['idx']} qid={x['question_id']} "
        f"tokens={x['token_count']} chars={x['char_count']} "
        f"dup4={x['duplicate_4gram_ratio']} top4_count={x['top_4gram_count']} "
        f"preview={x['text_preview']}"
    )
PY
}

cd "$PROJECT_ROOT"

IMAGENET_FIRST10="$OUT_DIR/ImageNet-R.first${NUM_SAMPLES}.json"
ARXIVQA_FIRST10="$OUT_DIR/ArxivQA.first${NUM_SAMPLES}.json"

make_first_n "$HARD_PATH/UCIT/ImageNet-R/test_3000.json" "$IMAGENET_FIRST10"
make_first_n "$HARD_PATH/UCIT/ArxivQA/test_3000.json" "$ARXIVQA_FIRST10"

echo "checkpoint=$CHECKPOINT"
echo "model_base=$MODEL_BASE"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "out_dir=$OUT_DIR"
echo "max_new_tokens=$MAX_NEW_TOKENS"

run_probe "task1_ImageNet-R" "$IMAGENET_FIRST10"
run_probe "task2_ArxivQA" "$ARXIVQA_FIRST10"

echo "Done. Inspect summaries under: $OUT_DIR"
