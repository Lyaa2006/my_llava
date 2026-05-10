#!/bin/bash
set -e

TASK_ID=$1

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
CONFIG_ROOT="${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}"

cd "$PROJECT_ROOT"

UCIT_SMOKE="${UCIT_SMOKE:-0}"
if [ "$UCIT_SMOKE" = "1" ]; then
    IMAGENET_DATA="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json"
    ARXIVQA_DATA="$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json"
    TASK1_EVAL="$CONFIG_ROOT/eval/task1_smoke.json"
    TASK2_EVAL="$CONFIG_ROOT/eval/task2_smoke.json"
else
    IMAGENET_DATA="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json"
    ARXIVQA_DATA="$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json"
    TASK1_EVAL="$CONFIG_ROOT/eval/task1.json"
    TASK2_EVAL="$CONFIG_ROOT/eval/task2.json"
fi

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK1_EVAL"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json "$IMAGENET_DATA" "$TASK2_EVAL"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json "$ARXIVQA_DATA" "$TASK2_EVAL"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/eval/task3.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/eval/task3.json
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/eval/task3.json
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/eval/task4.json
    
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/eval/task4.json
    
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/eval/task4.json
    
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $CONFIG_ROOT/eval/task4.json
elif [ "$TASK_ID" == "5" ]; then
    
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/eval/task5.json
    
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/eval/task5.json
    
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/eval/task5.json

    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $CONFIG_ROOT/eval/task5.json
    
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json $CONFIG_ROOT/eval/task5.json
else

    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json $CONFIG_ROOT/eval/task6.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json $CONFIG_ROOT/eval/task6.json
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/VizWiz.json $CONFIG_ROOT/eval/task6.json
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/IconQA.json $CONFIG_ROOT/eval/task6.json
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json $CONFIG_ROOT/eval/task6.json
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json $CONFIG_ROOT/eval/task6.json
fi
