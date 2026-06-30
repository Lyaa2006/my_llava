#!/bin/bash
set -euo pipefail

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
cd "$PROJECT_ROOT"

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_config_default() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    d = json.load(f)
print(d.get(key, default))
PY
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config_default "$TRAIN_CONFIG" rank 16)
EXPERT=$(read_config_default "$TRAIN_CONFIG" expert_num 6)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config_default "$TRAIN_CONFIG" previous_model "")
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
CUR_TASK=$(read_config_default "$TRAIN_CONFIG" cur_task 1)
LR=$(read_config_default "$TRAIN_CONFIG" lr 2e-4)
MODEL_MAX_LENGTH=$(read_config_default "$TRAIN_CONFIG" model_max_length 1024)
DESCRIPTION_PROMPT=${DESCRIPTION_PROMPT:-"Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations."}
DESCRIPTION_HIDDEN_LAYER=${DESCRIPTION_HIDDEN_LAYER:-"-2"}
DESCRIPTION_MAX_TOKENS=${DESCRIPTION_MAX_TOKENS:-"32"}
DESCRIPTION_CACHE_MODE=${DESCRIPTION_CACHE_MODE:-snapshot}
DESCRIPTION_CACHE_DIR=${DESCRIPTION_CACHE_DIR:-"$OUTPUT_DIR/description_cache_${DESCRIPTION_CACHE_MODE}"}

case "$DESCRIPTION_CACHE_MODE" in
    snapshot|static_backbone) ;;
    *)
        echo "ERROR: DESCRIPTION_CACHE_MODE must be 'snapshot' or 'static_backbone'." >&2
        exit 1
        ;;
esac

if [ -n "${UCIT_GPU_NUM:-}" ]; then
    GPU_NUM="$UCIT_GPU_NUM"
fi

AVAILABLE_GPU_NUM="$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(-1)
PY
)"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_GPU_NUM="$(python3 - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not value:
    print(0)
else:
    items = [x.strip() for x in value.split(",") if x.strip() != ""]
    print(len(items))
PY
)"
    if [ -n "${UCIT_GPU_NUM:-}" ] && [ "$CUDA_VISIBLE_GPU_NUM" -gt 0 ] && [ "$GPU_NUM" -gt "$CUDA_VISIBLE_GPU_NUM" ]; then
        echo "ERROR: UCIT_GPU_NUM=$GPU_NUM but CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" only exposes $CUDA_VISIBLE_GPU_NUM GPUs." >&2
        exit 1
    fi
else
    if [ "$AVAILABLE_GPU_NUM" -ge 0 ] && [ "$GPU_NUM" -gt "$AVAILABLE_GPU_NUM" ]; then
        echo "ERROR: UCIT_GPU_NUM=$GPU_NUM but torch.cuda.device_count()=$AVAILABLE_GPU_NUM. Check your GPU allocation." >&2
        exit 1
    fi
fi

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}

MASTER_PORT="${MASTER_PORT:-9001}"
CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-$((MASTER_PORT + 100))}"
CACHE_OUTPUT_DIR="${CACHE_OUTPUT_DIR:-/tmp/description_cache_only_${DESCRIPTION_CACHE_MODE}}"

ensure_free_port() {
    python3 - "$1" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, start + 200):
    sock = socket.socket()
    try:
        sock.bind(("", port))
        sock.close()
        print(port)
        raise SystemExit(0)
    except OSError:
        sock.close()
print(-1)
raise SystemExit(0)
PY
}

FREE_CACHE_MASTER_PORT="$(ensure_free_port "$CACHE_MASTER_PORT")"
if [ "$FREE_CACHE_MASTER_PORT" = "-1" ]; then
    echo "ERROR: No free cache master port found in range [$CACHE_MASTER_PORT, $((CACHE_MASTER_PORT + 199))]." >&2
    exit 1
fi
CACHE_MASTER_PORT="$FREE_CACHE_MASTER_PORT"

DS_INCLUDE_ARGS="--include localhost:$GPU_LIST"

COMMON_ARGS=(
    --deepspeed ./scripts/zero2.json
    --expert_num "$EXPERT"
    --model_name_or_path "$MODEL_NAME"
    --version "$PROMPT_VERSION"
    --data_path "$DATA_PATH"
    --image_folder "$IMAGE"
    --vision_tower "$VISION_TOWER"
    --text_tower "$VISION_TOWER"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --output_dir "$CACHE_OUTPUT_DIR"
    --cur_task "$CUR_TASK"
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 1
    --evaluation_strategy no
    --save_strategy no
    --learning_rate "$LR"
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length "$MODEL_MAX_LENGTH"
    --gradient_checkpointing False
    --dataloader_num_workers 0
    --lazy_preprocess True
    --extract_description_cache_only True
    --description_cache_dir "$DESCRIPTION_CACHE_DIR"
    --description_prompt "$DESCRIPTION_PROMPT"
    --description_hidden_layer "$DESCRIPTION_HIDDEN_LAYER"
    --description_max_tokens "$DESCRIPTION_MAX_TOKENS"
    --description_cache_generation_mode "$DESCRIPTION_CACHE_MODE"
    --report_to none
)

if [ "$DESCRIPTION_CACHE_MODE" = "snapshot" ]; then
    if [ -z "$PREVIOUS" ]; then
        echo "ERROR: snapshot cache generation requires previous_model in the train config." >&2
        exit 1
    fi
    deepspeed $DS_INCLUDE_ARGS --master_port "$CACHE_MASTER_PORT" llava/train/train_mem_MOE.py \
        --lora_enable True --lora_r "$RANK" --lora_alpha "$((RANK * 2))" --mm_projector_lr 2e-5 \
        --previous_task_model_path "$PREVIOUS" \
        "${COMMON_ARGS[@]}"
else
    deepspeed $DS_INCLUDE_ARGS --master_port "$CACHE_MASTER_PORT" llava/train/train_mem_MOE.py \
        --lora_enable False \
        "${COMMON_ARGS[@]}"
fi
