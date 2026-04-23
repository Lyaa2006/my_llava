#!/bin/bash
set -e

HARD_PATH=/mnt/lyaa/MCITlib
CONFIG_ROOT=${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}

pip install -e .

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
    $CONFIG_ROOT/train/task1_smoke_loraftlen.json

bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
    $CONFIG_ROOT/eval/task1_smoke.json

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json \
    $CONFIG_ROOT/train/task2_smoke_loraftlen.json

bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
    $CONFIG_ROOT/eval/task2_smoke.json
