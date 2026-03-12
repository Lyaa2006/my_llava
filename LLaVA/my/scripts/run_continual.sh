#!/bin/bash

# 设置环境
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据你的GPU数量调整
export PYTHONPATH=/mnt/lyaa/MCITlib/LLaVA/my:$PYTHONPATH

# 训练参数
BASE_MODEL_PATH="/mnt/lyaa/MCITlib/llava-v1.5-7b"
VISION_TOWER_PATH="/mnt/lyaa/MCITlib/clip-vit-large-patch14-336"
OUTPUT_BASE="/mnt/lyaa/MCITlib/checkpoints/continual"

# LoRA参数
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj v_proj"

# Loss权重
LOSS_HIDDEN=0.2      # hidden state一致性
LOSS_TEXT_QA=0.3     # 纯文本QA（description+question）
LOSS_VISION_QA=0.5   # 多模态QA

# 训练参数
BATCH_SIZE=4
GRAD_ACCUM=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3
SAVE_STEPS=500

# 运行训练
python train_continual.py \
    --output_dir /mnt/lyaa/MCITlib/checkpoints/continual \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --learning_rate 2e-4