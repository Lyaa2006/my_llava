#!/bin/bash

# 1. 环境基础配置
export PYTHONPATH=/mnt/lyaa/MCITlib/LLaVA/my:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 缓解显存碎片化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# 2. 路径配置
BASE_MODEL_PATH="/mnt/lyaa/MCITlib/llava-v1.5-7b"
VISION_TOWER_PATH="/mnt/lyaa/MCITlib/clip-vit-large-patch14-336"
OUTPUT_DIR="/mnt/lyaa/MCITlib/checkpoints/continual/initial_run"
CACHE_DIR="/mnt/lyaa/MCITlib/description_cache"

# 3. 运行训练 (强制使用单卡模式以规避 DataParallel 报错)
# 注意：如果你想用多卡，必须配合 accelerate launch 或 deepspeed，否则 DP 会导致 OOM
CUDA_VISIBLE_DEVICES=0 python train_continual.py \
    --model_name_or_path "$BASE_MODEL_PATH" \
    --vision_tower "$VISION_TOWER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --description_cache_dir "$CACHE_DIR" \
    --fp16 True \
    --tf32 True \
    --bf16 False \
    --model_max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "none" \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16 \
    --loss_hidden_weight 0.2 \
    --loss_text_qa_weight 0.3 \
    --loss_vision_qa_weight 0.5 \
    --hidden_layer_idx -10