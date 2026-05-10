#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
CONFIG_ROOT="${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}"

cd "$PROJECT_ROOT"

export UCIT_RUN_ID="${UCIT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export DESCRIPTION_MAX_TOKENS="${DESCRIPTION_MAX_TOKENS:-8}"
export DESCRIPTION_UTILITY_WEIGHT="${DESCRIPTION_UTILITY_WEIGHT:-0.2}"
export ORTH_LORA_WEIGHT="${ORTH_LORA_WEIGHT:-0.1}"
LOG_DIR="${LOG_DIR:-/mnt/lyaa/my_llava/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/smoke_UCIT_${UCIT_RUN_ID}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "UCIT smoke run id: $UCIT_RUN_ID"

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

echo "========== smoke train task1 =========="
bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json \
    $CONFIG_ROOT/train/task1_smoke.json

echo "========== smoke eval task1 =========="
bash scripts/MCITlib/Eval_UCIT/Eval_finetune_smoke.sh 1

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

echo "========== smoke train task2 =========="
bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json \
    $CONFIG_ROOT/train/task2_smoke.json

echo "========== smoke eval task2 =========="
bash scripts/MCITlib/Eval_UCIT/Eval_finetune_smoke.sh 2
