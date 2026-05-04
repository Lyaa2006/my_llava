#!/bin/bash
set -e

HARD_PATH=/mnt/lyaa/my_llava
CONFIG_ROOT=${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}
DESCRIPTION_MAX_TOKENS=${DESCRIPTION_MAX_TOKENS:-8}
DESCRIPTION_UTILITY_WEIGHT=${DESCRIPTION_UTILITY_WEIGHT:-0.2}
export UCIT_RUN_ID="${UCIT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
    $CONFIG_ROOT/train/task1_smoke.json

bash scripts/MCITlib/Eval_UCIT/Eval_finetune_smoke.sh 1

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json \
    $CONFIG_ROOT/train/task2_smoke.json

bash scripts/MCITlib/Eval_UCIT/Eval_finetune_smoke.sh 2
