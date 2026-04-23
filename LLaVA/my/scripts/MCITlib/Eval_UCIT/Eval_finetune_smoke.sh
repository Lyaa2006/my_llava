#!/bin/bash
set -e

TASK_ID=$1
HARD_PATH=/mnt/lyaa/MCITlib
CONFIG_ROOT=${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
        $HARD_PATH/configs/modal_configs/llava.json \
        $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
        $CONFIG_ROOT/eval/task1_smoke.json
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
        $HARD_PATH/configs/modal_configs/llava.json \
        $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
        $CONFIG_ROOT/eval/task2_smoke.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh \
        $HARD_PATH/configs/modal_configs/llava.json \
        $HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json \
        $CONFIG_ROOT/eval/task2_smoke.json
else
    echo "Unsupported smoke TASK_ID: $TASK_ID" >&2
    exit 1
fi
