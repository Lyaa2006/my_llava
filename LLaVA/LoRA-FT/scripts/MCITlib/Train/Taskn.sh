#!/bin/bash
set -e

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_optional_config() {
    python3 - "$1" "$2" "$3" <<'PY'
import json, sys
path, key, default = sys.argv[1:]
with open(path, "r") as f:
    data = json.load(f)
value = data.get(key, default)
if isinstance(value, bool):
    print("True" if value else "False")
else:
    print(value)
PY
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
SAVE_STEPS=$(read_optional_config "$TRAIN_CONFIG" save_steps 50000)
MAX_STEPS=$(read_optional_config "$TRAIN_CONFIG" max_steps -1)
DATALOADER_NUM_WORKERS=$(read_optional_config "$TRAIN_CONFIG" dataloader_num_workers 4)
MODEL_MAX_LENGTH=$(read_optional_config "$TRAIN_CONFIG" model_max_length 2048)
BF16=$(read_optional_config "$TRAIN_CONFIG" bf16 True)
FP16=$(read_optional_config "$TRAIN_CONFIG" fp16 False)
TF32=$(read_optional_config "$TRAIN_CONFIG" tf32 True)
BITS=$(read_optional_config "$TRAIN_CONFIG" bits 16)
MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}

DEEPSPEED_PREFIX=()
DEEPSPEED_ARGS=(deepspeed)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEEPSPEED_PREFIX=(env -u CUDA_VISIBLE_DEVICES)
    DEEPSPEED_ARGS+=(--include "localhost:${CUDA_VISIBLE_DEVICES}")
else
    GPU_LIST=""
    for i in $(seq 0 $((GPU_NUM-1))); do
        GPU_LIST+="$i,"
    done
    GPU_LIST=${GPU_LIST%,}
    DEEPSPEED_ARGS+=(--include "localhost:$GPU_LIST")
fi

DEEPSPEED_ARGS+=(--master_port "$MASTER_PORT")

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

"${DEEPSPEED_PREFIX[@]}" "${DEEPSPEED_ARGS[@]}" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_NAME \
    --previous_task_model_path $PREVIOUS \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bits $BITS \
    --bf16 $BF16 \
    --fp16 $FP16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --max_steps $MAX_STEPS \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 $TF32 \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --report_to none
