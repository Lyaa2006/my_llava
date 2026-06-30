#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <snapshot|static_backbone>" >&2
    exit 1
fi

CACHE_MODE="$1"
case "$CACHE_MODE" in
    snapshot|static_backbone) ;;
    *)
        echo "ERROR: cache mode must be 'snapshot' or 'static_backbone'." >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"
HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export UCIT_GPU_NUM="${UCIT_GPU_NUM:-4}"

unique_suffix() {
    date +%Y%m%d_%H%M%S_%N
}

uniquify_dir_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        printf '%s\n' "$path"
        return
    fi
    printf '%s__%s\n' "$path" "$(unique_suffix)"
}

uniquify_file_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        printf '%s\n' "$path"
        return
    fi
    case "$path" in
        *.log) printf '%s__%s.log\n' "${path%.log}" "$(unique_suffix)" ;;
        *) printf '%s__%s\n' "$path" "$(unique_suffix)" ;;
    esac
}

RUN_TAG="${RUN_TAG:-task2_last1_${CACHE_MODE}_$(unique_suffix)}"
OUTPUT_DIR_DEFAULT="$HARD_PATH/checkpoint/UCIT/LLaVA-1.5/my_method2/cache_ablation/${RUN_TAG}"
CACHE_DIR_DEFAULT="$OUTPUT_DIR_DEFAULT/description_cache"
LOG_DIR="${LOG_DIR:-$HARD_PATH/logs/task2_cache_ablation}"
mkdir -p "$LOG_DIR"
LOG_FILE_DEFAULT="$LOG_DIR/${RUN_TAG}.log"

OUTPUT_DIR="$(uniquify_dir_path "${OUTPUT_DIR:-$OUTPUT_DIR_DEFAULT}")"
CACHE_DIR="$(uniquify_dir_path "${CACHE_DIR:-$CACHE_DIR_DEFAULT}")"
LOG_FILE="$(uniquify_file_path "${LOG_FILE:-$LOG_FILE_DEFAULT}")"

export UCIT_TASK_OUTPUT_DIR_OVERRIDE="$OUTPUT_DIR"
export DESCRIPTION_CACHE_DIR="$CACHE_DIR"
export DESCRIPTION_EXTRACT_CACHE=1
export DESCRIPTION_CACHE_MODE="$CACHE_MODE"

# Keep the previous weighting setup; only change to exclude only the last layer
# from the auxiliary description-alignment constraint.
export DESCRIPTION_ALIGN_WEIGHT="${DESCRIPTION_ALIGN_WEIGHT:-0.1}"
export DESCRIPTION_UTILITY_WEIGHT="${DESCRIPTION_UTILITY_WEIGHT:-0.0}"
export STANDARD_CE_WEIGHT="${STANDARD_CE_WEIGHT:-1.0}"
export ENABLE_LAYERWISE_AUX_LOSS="${ENABLE_LAYERWISE_AUX_LOSS:-True}"
export AUX_EXCLUDE_LAST_N_LAYERS="${AUX_EXCLUDE_LAST_N_LAYERS:-1}"
export DESCRIPTION_ALIGN_LOSS_CAP="${DESCRIPTION_ALIGN_LOSS_CAP:-2.0}"
export DESCRIPTION_UTILITY_LOSS_CAP="${DESCRIPTION_UTILITY_LOSS_CAP:-2.0}"

export SKIP_BAD_BATCHES="${SKIP_BAD_BATCHES:-True}"
export MAX_CONSECUTIVE_BAD_BATCHES="${MAX_CONSECUTIVE_BAD_BATCHES:-4000}"
export DEBUG_FORCE_BAD_BATCH_STEP="${DEBUG_FORCE_BAD_BATCH_STEP:--1}"

export MASTER_PORT="${MASTER_PORT:-29501}"
export CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-29601}"

{
    echo "Logging to: $LOG_FILE"
    echo "Project root: $PROJECT_ROOT"
    echo "Cache mode: $CACHE_MODE"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "UCIT_GPU_NUM: $UCIT_GPU_NUM"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "DESCRIPTION_CACHE_DIR: $CACHE_DIR"
    echo "DESCRIPTION_ALIGN_WEIGHT: $DESCRIPTION_ALIGN_WEIGHT"
    echo "DESCRIPTION_UTILITY_WEIGHT: $DESCRIPTION_UTILITY_WEIGHT"
    echo "STANDARD_CE_WEIGHT: $STANDARD_CE_WEIGHT"
    echo "AUX_EXCLUDE_LAST_N_LAYERS: $AUX_EXCLUDE_LAST_N_LAYERS"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "CACHE_MASTER_PORT: $CACHE_MASTER_PORT"
} | tee "$LOG_FILE"

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/modal_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json" \
    "$HARD_PATH/configs/train_configs/my_method2/LLaVA/UCIT/train/task2.json" \
    2>&1 | tee -a "$LOG_FILE"
