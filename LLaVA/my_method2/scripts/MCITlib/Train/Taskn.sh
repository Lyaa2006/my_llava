#!/bin/bash

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
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS="${PREVIOUS:-$(read_config "$TRAIN_CONFIG" previous_model)}"
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
if [ -n "${UCIT_TASK_OUTPUT_DIR_OVERRIDE:-}" ]; then
    OUTPUT_DIR="$UCIT_TASK_OUTPUT_DIR_OVERRIDE"
elif [ -n "${UCIT_OUTPUT_DIR_OVERRIDE:-}" ]; then
    OUTPUT_DIR="$UCIT_OUTPUT_DIR_OVERRIDE"
fi
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
MAX_STEPS=$(read_config_default "$TRAIN_CONFIG" max_steps -1)
SAVE_STEPS=$(read_config_default "$TRAIN_CONFIG" save_steps 50000)
MODEL_MAX_LENGTH=$(read_config_default "$TRAIN_CONFIG" model_max_length 2048)
DATALOADER_NUM_WORKERS=$(read_config_default "$TRAIN_CONFIG" dataloader_num_workers 4)

if [ -n "${UCIT_MAX_STEPS:-}" ]; then
    MAX_STEPS="$UCIT_MAX_STEPS"
fi

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

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

DESCRIPTION_PROMPT=${DESCRIPTION_PROMPT:-"Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations."}
DESCRIPTION_HIDDEN_LAYER=${DESCRIPTION_HIDDEN_LAYER:-"-2"}
DESCRIPTION_MAX_TOKENS=${DESCRIPTION_MAX_TOKENS:-"32"}
DESCRIPTION_ALIGN_WEIGHT=${DESCRIPTION_ALIGN_WEIGHT:-"1.0"}
DESCRIPTION_UTILITY_WEIGHT=${DESCRIPTION_UTILITY_WEIGHT:-"1.0"}
STANDARD_CE_WEIGHT=${STANDARD_CE_WEIGHT:-"1.0"}
ENABLE_LAYERWISE_AUX_LOSS=${ENABLE_LAYERWISE_AUX_LOSS:-"False"}
AUX_EXCLUDE_LAST_N_LAYERS=${AUX_EXCLUDE_LAST_N_LAYERS:-"4"}
DESCRIPTION_ALIGN_LOSS_CAP=${DESCRIPTION_ALIGN_LOSS_CAP:-"-1.0"}
DESCRIPTION_UTILITY_LOSS_CAP=${DESCRIPTION_UTILITY_LOSS_CAP:-"-1.0"}
DESCRIPTION_CACHE_DIR=${DESCRIPTION_CACHE_DIR:-"$OUTPUT_DIR/description_cache"}
DESCRIPTION_EXTRACT_CACHE=${DESCRIPTION_EXTRACT_CACHE:-"0"}
DESCRIPTION_CACHE_MODE=${DESCRIPTION_CACHE_MODE:-"snapshot"}
SKIP_BAD_BATCHES=${SKIP_BAD_BATCHES:-"True"}
MAX_CONSECUTIVE_BAD_BATCHES=${MAX_CONSECUTIVE_BAD_BATCHES:-"200"}
DEBUG_FORCE_BAD_BATCH_STEP=${DEBUG_FORCE_BAD_BATCH_STEP:-"-1"}

case "$DESCRIPTION_CACHE_MODE" in
    snapshot|static_backbone) ;;
    *)
        echo "ERROR: DESCRIPTION_CACHE_MODE must be 'snapshot' or 'static_backbone'." >&2
        exit 1
        ;;
esac

EXTRA_ARGS=""
if [ "$MAX_STEPS" -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
fi

DS_INCLUDE_ARGS=""
DS_INCLUDE_ARGS="--include localhost:$GPU_LIST"

if [ "$DESCRIPTION_EXTRACT_CACHE" = "1" ]; then
    CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-$((MASTER_PORT + 100))}"
    if [ "$DESCRIPTION_CACHE_MODE" = "snapshot" ]; then
        deepspeed $DS_INCLUDE_ARGS --master_port "$CACHE_MASTER_PORT" llava/train/train_mem_MOE.py \
            --deepspeed ./scripts/zero2.json \
            --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 2e-5 \
            --expert_num $EXPERT \
            --model_name_or_path $MODEL_NAME \
            --previous_task_model_path $PREVIOUS \
            --version $PROMPT_VERSION \
            --data_path $DATA_PATH \
            --image_folder $IMAGE \
            --vision_tower $VISION_TOWER \
            --text_tower $VISION_TOWER \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir /tmp/description_cache_only \
            --cur_task $CUR_TASK \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate $LR \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length $MODEL_MAX_LENGTH \
            --gradient_checkpointing False \
            --dataloader_num_workers 0 \
            --lazy_preprocess True \
            --extract_description_cache_only True \
            --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
            --description_prompt "$DESCRIPTION_PROMPT" \
            --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
            --description_max_tokens $DESCRIPTION_MAX_TOKENS \
            --description_cache_generation_mode "$DESCRIPTION_CACHE_MODE" \
            --report_to none \
            $EXTRA_ARGS
    else
        deepspeed $DS_INCLUDE_ARGS --master_port "$CACHE_MASTER_PORT" llava/train/train_mem_MOE.py \
            --deepspeed ./scripts/zero2.json \
            --lora_enable False \
            --expert_num $EXPERT \
            --model_name_or_path $MODEL_NAME \
            --version $PROMPT_VERSION \
            --data_path $DATA_PATH \
            --image_folder $IMAGE \
            --vision_tower $VISION_TOWER \
            --text_tower $VISION_TOWER \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir /tmp/description_cache_only \
            --cur_task $CUR_TASK \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate $LR \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length $MODEL_MAX_LENGTH \
            --gradient_checkpointing False \
            --dataloader_num_workers 0 \
            --lazy_preprocess True \
            --extract_description_cache_only True \
            --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
            --description_prompt "$DESCRIPTION_PROMPT" \
            --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
            --description_max_tokens $DESCRIPTION_MAX_TOKENS \
            --description_cache_generation_mode "$DESCRIPTION_CACHE_MODE" \
            --report_to none \
            $EXTRA_ARGS
    fi
fi

deepspeed $DS_INCLUDE_ARGS --master_port "$MASTER_PORT" llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 2e-5 \
    --expert_num $EXPERT \
    --model_name_or_path $MODEL_NAME \
    --previous_task_model_path $PREVIOUS \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --text_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --cur_task $CUR_TASK \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --enable_description_cl True \
    --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
    --description_prompt "$DESCRIPTION_PROMPT" \
    --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
    --description_max_tokens $DESCRIPTION_MAX_TOKENS \
    --description_align_weight $DESCRIPTION_ALIGN_WEIGHT \
    --description_utility_weight $DESCRIPTION_UTILITY_WEIGHT \
    --standard_ce_weight $STANDARD_CE_WEIGHT \
    --enable_layerwise_aux_loss $ENABLE_LAYERWISE_AUX_LOSS \
    --aux_exclude_last_n_layers $AUX_EXCLUDE_LAST_N_LAYERS \
    --description_align_loss_cap $DESCRIPTION_ALIGN_LOSS_CAP \
    --description_utility_loss_cap $DESCRIPTION_UTILITY_LOSS_CAP \
    --skip_bad_batches $SKIP_BAD_BATCHES \
    --max_consecutive_bad_batches $MAX_CONSECUTIVE_BAD_BATCHES \
    --debug_force_bad_batch_step $DEBUG_FORCE_BAD_BATCH_STEP \
    --report_to none \
    $EXTRA_ARGS
