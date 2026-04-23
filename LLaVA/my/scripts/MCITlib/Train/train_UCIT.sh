#!/bin/bash
set -e

HARD_PATH=/mnt/lyaa/MCITlib
CONFIG_ROOT=${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $CONFIG_ROOT/train/task1.json

bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $CONFIG_ROOT/train/task2.json

bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2
