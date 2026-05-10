#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
CONFIG_ROOT="${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/UCIT}"

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

export UCIT_RUN_ID="${UCIT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
UCIT_SMOKE="${UCIT_SMOKE:-0}"
LOG_DIR="${LOG_DIR:-/mnt/lyaa/my_llava/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_UCIT_${UCIT_RUN_ID}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "UCIT run id: $UCIT_RUN_ID"

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

if [ "$UCIT_SMOKE" = "1" ]; then
    DATA_TASK1="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json"
    TRAIN_TASK1="$CONFIG_ROOT/train/task1_smoke.json"
    DATA_TASK2="$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json"
    TRAIN_TASK2="$CONFIG_ROOT/train/task2_smoke.json"
else
    DATA_TASK1="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json"
    TRAIN_TASK1="$CONFIG_ROOT/train/task1.json"
    DATA_TASK2="$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json"
    TRAIN_TASK2="$CONFIG_ROOT/train/task2.json"
fi

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $DATA_TASK1 \
    $TRAIN_TASK1

bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install -e . --no-build-isolation
fi

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/modal_configs/llava.json \
    $DATA_TASK2 \
    $TRAIN_TASK2

bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2
