#!/bin/bash
set -e

TASK_ID=$1

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
CONFIG_ROOT="${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT}"

cd "$PROJECT_ROOT"

UCIT_SMOKE="${UCIT_SMOKE:-0}"
UCIT_SKIP_VIZWIZ="${UCIT_SKIP_VIZWIZ:-0}"
UCIT_SKIP_FLICKR="${UCIT_SKIP_FLICKR:-0}"
if [ "$UCIT_SMOKE" = "1" ]; then
    export UCIT_MAX_SAMPLES="${UCIT_MAX_SAMPLES:-32}"
    IMAGENET_DATA="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json"
    ARXIVQA_DATA="$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json"
    VIZWIZ_DATA="$HARD_PATH/configs/data_configs/UCIT/VizWiz.json"
    ICONQA_DATA="$HARD_PATH/configs/data_configs/UCIT/IconQA.json"
    CLEVR_DATA="$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json"
    FLICKR_DATA="$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json"
    TASK1_EVAL="$CONFIG_ROOT/eval/task1_smoke.json"
    TASK2_EVAL="$CONFIG_ROOT/eval/task2_smoke.json"
    TASK3_EVAL="$CONFIG_ROOT/eval/task3_smoke.json"
    TASK4_EVAL="$CONFIG_ROOT/eval/task4_smoke.json"
    TASK5_EVAL="$CONFIG_ROOT/eval/task5_smoke.json"
    TASK6_EVAL="$CONFIG_ROOT/eval/task6_smoke.json"
else
    IMAGENET_DATA="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json"
    ARXIVQA_DATA="$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json"
    VIZWIZ_DATA="$HARD_PATH/configs/data_configs/UCIT/VizWiz.json"
    ICONQA_DATA="$HARD_PATH/configs/data_configs/UCIT/IconQA.json"
    CLEVR_DATA="$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json"
    FLICKR_DATA="$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json"
    TASK1_EVAL="$CONFIG_ROOT/eval/task1.json"
    TASK2_EVAL="$CONFIG_ROOT/eval/task2.json"
    TASK3_EVAL="$CONFIG_ROOT/eval/task3.json"
    TASK4_EVAL="$CONFIG_ROOT/eval/task4.json"
    TASK5_EVAL="$CONFIG_ROOT/eval/task5.json"
    TASK6_EVAL="$CONFIG_ROOT/eval/task6.json"
fi

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK1_EVAL"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK2_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ARXIVQA_DATA" "$TASK2_EVAL"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK3_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ARXIVQA_DATA" "$TASK3_EVAL"
    if [ "$UCIT_SKIP_VIZWIZ" != "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json "$VIZWIZ_DATA" "$TASK3_EVAL"
    fi
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK4_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ARXIVQA_DATA" "$TASK4_EVAL"
    if [ "$UCIT_SKIP_VIZWIZ" != "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json "$VIZWIZ_DATA" "$TASK4_EVAL"
    fi
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ICONQA_DATA" "$TASK4_EVAL"
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ARXIVQA_DATA" "$TASK5_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK5_EVAL"
    if [ "$UCIT_SKIP_VIZWIZ" != "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json "$VIZWIZ_DATA" "$TASK5_EVAL"
    fi
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ICONQA_DATA" "$TASK5_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/modal_configs/llava.json "$CLEVR_DATA" "$TASK5_EVAL"
else
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK6_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ARXIVQA_DATA" "$TASK6_EVAL"
    if [ "$UCIT_SKIP_VIZWIZ" != "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json "$VIZWIZ_DATA" "$TASK6_EVAL"
    fi
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ICONQA_DATA" "$TASK6_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/modal_configs/llava.json "$CLEVR_DATA" "$TASK6_EVAL"
    if [ "$UCIT_SKIP_FLICKR" != "1" ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $HARD_PATH/configs/modal_configs/llava.json "$FLICKR_DATA" "$TASK6_EVAL"
    fi
fi
