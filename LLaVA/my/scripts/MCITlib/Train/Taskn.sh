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

count_expected_cache_entries() {
    python3 - "$1" <<'PY'
import json, sys
with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(sum(1 for sample in data if "image" in sample))
PY
}

cache_meta_matches() {
    python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, os, sys
cache_dir, data_path, prompt, hidden_layer, max_tokens = sys.argv[1:]
meta_path = os.path.join(cache_dir, "meta.json")
if not os.path.exists(meta_path):
    print("False")
    raise SystemExit
with open(meta_path, "r") as f:
    meta = json.load(f)
ok = (
    meta.get("data_path") == data_path
    and meta.get("description_prompt") == prompt
    and str(meta.get("description_hidden_layer")) == str(hidden_layer)
    and str(meta.get("description_max_tokens")) == str(max_tokens)
)
print("True" if ok else "False")
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
RUN_SUFFIX=${UCIT_RUN_ID:+_$UCIT_RUN_ID}
OUTPUT_DIR="${OUTPUT_DIR}${RUN_SUFFIX}"
PREVIOUS="${PREVIOUS}${RUN_SUFFIX}"
DESCRIPTION_PROMPT=${DESCRIPTION_PROMPT:-"Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations."}
DESCRIPTION_HIDDEN_LAYER=${DESCRIPTION_HIDDEN_LAYER:-"-2"}
DESCRIPTION_MAX_TOKENS=${DESCRIPTION_MAX_TOKENS:-"32"}
DESCRIPTION_ALIGN_WEIGHT=${DESCRIPTION_ALIGN_WEIGHT:-"1.0"}
DESCRIPTION_UTILITY_WEIGHT=${DESCRIPTION_UTILITY_WEIGHT:-"1.0"}
STANDARD_CE_WEIGHT=${STANDARD_CE_WEIGHT:-"1.0"}
ORTH_LORA_WEIGHT=${ORTH_LORA_WEIGHT:-"0.0"}
OLD_LORA_SCALE=${OLD_LORA_SCALE:-"1.0"}
DESCRIPTION_CACHE_DIR=${DESCRIPTION_CACHE_DIR:-"$OUTPUT_DIR/reference_description_cache"}
FREEZE_MM_PROJECTOR=${FREEZE_MM_PROJECTOR:-"0"}
MAX_STEPS=$(read_optional_config "$TRAIN_CONFIG" max_steps -1)
SAVE_STEPS=$(read_optional_config "$TRAIN_CONFIG" save_steps 50000)
DATALOADER_NUM_WORKERS=$(read_optional_config "$TRAIN_CONFIG" dataloader_num_workers 4)
MODEL_MAX_LENGTH=$(read_optional_config "$TRAIN_CONFIG" model_max_length 2048)
BF16=$(read_optional_config "$TRAIN_CONFIG" bf16 True)
FP16=$(read_optional_config "$TRAIN_CONFIG" fp16 False)
TF32=$(read_optional_config "$TRAIN_CONFIG" tf32 True)
BITS=$(read_optional_config "$TRAIN_CONFIG" bits 16)
MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
EXPECTED_CACHE_ENTRIES=$(count_expected_cache_entries "$DATA_PATH")
CACHE_READY=False

if [ -d "$DESCRIPTION_CACHE_DIR" ]; then
    EXISTING_CACHE_ENTRIES=$(find "$DESCRIPTION_CACHE_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    CACHE_META_READY=$(cache_meta_matches \
        "$DESCRIPTION_CACHE_DIR" \
        "$DATA_PATH" \
        "$DESCRIPTION_PROMPT" \
        "$DESCRIPTION_HIDDEN_LAYER" \
        "$DESCRIPTION_MAX_TOKENS")
    if [ "$EXISTING_CACHE_ENTRIES" -ge "$EXPECTED_CACHE_ENTRIES" ] && [ "$EXPECTED_CACHE_ENTRIES" -gt 0 ] && [ "$CACHE_META_READY" = "True" ]; then
        CACHE_READY=True
    elif [ "$EXISTING_CACHE_ENTRIES" -gt 0 ]; then
        echo "Description cache exists but metadata does not match current prompt/settings; rebuilding: $DESCRIPTION_CACHE_DIR"
    fi
fi

DEEPSPEED_PREFIX=()
DEEPSPEED_ARGS=(deepspeed)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # DeepSpeed auto-translates CUDA_VISIBLE_DEVICES into --include and can
    # mis-handle non-zero physical ids like 7,8. Pass the include list
    # ourselves and clear CUDA_VISIBLE_DEVICES only for the launcher.
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

if [ "$CACHE_READY" != "True" ]; then
    rm -rf "$DESCRIPTION_CACHE_DIR"
    python llava/train/train.py \
        --lora_enable True \
        --lora_r $RANK \
        --lora_alpha $((RANK * 2)) \
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
        --bits $BITS \
        --bf16 $BF16 \
        --fp16 $FP16 \
        --output_dir $OUTPUT_DIR \
        --model_max_length $MODEL_MAX_LENGTH \
        --lazy_preprocess True \
        --description_prompt "$DESCRIPTION_PROMPT" \
        --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
        --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
        --description_max_tokens $DESCRIPTION_MAX_TOKENS \
        --extract_description_cache_only True
fi

FREEZE_MM_ARGS=()
if [ "$FREEZE_MM_PROJECTOR" = "1" ]; then
    FREEZE_MM_ARGS+=(--freeze_mm_mlp_adapter True)
fi

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
    --description_prompt "$DESCRIPTION_PROMPT" \
    --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
    --enable_description_cl True \
    --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
    --description_max_tokens $DESCRIPTION_MAX_TOKENS \
    --description_align_weight $DESCRIPTION_ALIGN_WEIGHT \
    --description_utility_weight $DESCRIPTION_UTILITY_WEIGHT \
    --standard_ce_weight $STANDARD_CE_WEIGHT \
    --orth_lora_weight $ORTH_LORA_WEIGHT \
    --old_lora_scale $OLD_LORA_SCALE \
    "${FREEZE_MM_ARGS[@]}" \
    --report_to none
