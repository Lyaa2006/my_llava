#!/bin/bash
set -e

HARD_PATH=/mnt/lyaa/MCITlib
CONFIG_ROOT=${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/MLLM-DCL}

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json \
    $CONFIG_ROOT/train/task1.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 1 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json \
    $CONFIG_ROOT/train/task2.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 2 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json \
    $CONFIG_ROOT/train/task3.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 3 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json \
    $CONFIG_ROOT/train/task4.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 4 $HARD_PATH

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json \
    $CONFIG_ROOT/train/task5.json
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 5 $HARD_PATH
