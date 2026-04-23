#!/bin/bash

HARD_PATH=/mnt/lyaa/MCITlib
RUN_EVAL=${RUN_EVAL:-0}

pip install -e .
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT/train/task1.json

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT/train/task2.json

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT/train/task3.json

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT/train/task4.json

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT/train/task5.json

pip install -e .
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT/train/task6.json

if [ "$RUN_EVAL" = "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6
fi

