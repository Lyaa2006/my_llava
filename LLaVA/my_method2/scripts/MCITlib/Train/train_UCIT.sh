#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT_DEFAULT="$(realpath "$PROJECT_ROOT/../..")"

HARD_PATH="${HARD_PATH:-${MCITLIB_ROOT:-$MCITLIB_ROOT_DEFAULT}}"
TRAIN_CONFIG_ROOT="${TRAIN_CONFIG_ROOT:-${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/my_method2/LLaVA/UCIT}}"
EVAL_CONFIG_ROOT="${UCIT_EVAL_CONFIG_ROOT:-$HARD_PATH/configs/train_configs/HiDe/LLaVA/UCIT}"
CONFIG_ROOT="$TRAIN_CONFIG_ROOT"
export HARD_PATH

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

export UCIT_RUN_ID="${UCIT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
UCIT_SMOKE="${UCIT_SMOKE:-0}"
UCIT_TASK_SET="${UCIT_TASK_SET:-first2}"
UCIT_START_TASK="${UCIT_START_TASK:-1}"
UCIT_END_TASK="${UCIT_END_TASK:-}"
UCIT_ONLY="${UCIT_ONLY:-all}"
UCIT_TASKS="${UCIT_TASKS:-}"
UCIT_EVAL_MODE="${UCIT_EVAL_MODE:-}"
UCIT_REINSTALL_BETWEEN_TASKS="${UCIT_REINSTALL_BETWEEN_TASKS:-}"
UCIT_SKIP_VIZWIZ="${UCIT_SKIP_VIZWIZ:-}"
UCIT_SKIP_FLICKR="${UCIT_SKIP_FLICKR:-}"

if [ -z "$UCIT_END_TASK" ]; then
    if [ "$UCIT_TASK_SET" = "all" ]; then
        UCIT_END_TASK=6
    else
        UCIT_END_TASK=2
    fi
fi
if [ "$UCIT_SMOKE" = "1" ]; then
    export UCIT_MAX_TRAIN_SAMPLES="${UCIT_MAX_TRAIN_SAMPLES:-8}"
    export UCIT_MAX_SAMPLES="${UCIT_MAX_SAMPLES:-8}"
    export UCIT_MAX_NEW_TOKENS="${UCIT_MAX_NEW_TOKENS:-64}"
    if [ -n "${UCIT_SMOKE_TASKS:-}" ]; then
        UCIT_TASKS="$UCIT_SMOKE_TASKS"
    elif [ -z "$UCIT_TASKS" ] && [ "$UCIT_TASK_SET" != "all" ]; then
        UCIT_TASKS="1,2"
    fi
    UCIT_EVAL_MODE="${UCIT_EVAL_MODE:-final}"
    UCIT_REINSTALL_BETWEEN_TASKS="${UCIT_REINSTALL_BETWEEN_TASKS:-0}"
    UCIT_SKIP_VIZWIZ="${UCIT_SKIP_VIZWIZ:-1}"
    UCIT_SKIP_FLICKR="${UCIT_SKIP_FLICKR:-1}"
else
    UCIT_EVAL_MODE="${UCIT_EVAL_MODE:-per_task}"
    UCIT_REINSTALL_BETWEEN_TASKS="${UCIT_REINSTALL_BETWEEN_TASKS:-1}"
    UCIT_SKIP_VIZWIZ="${UCIT_SKIP_VIZWIZ:-0}"
    UCIT_SKIP_FLICKR="${UCIT_SKIP_FLICKR:-0}"
fi
export UCIT_SKIP_VIZWIZ
export UCIT_SKIP_FLICKR
export MASTER_PORT="${MASTER_PORT:-$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

ensure_free_master_port() {
    local start_port="$1"
    python3 - "$start_port" <<'PY'
import socket, sys
start = int(sys.argv[1])
port = start
while port < start + 200:
    s = socket.socket()
    try:
        s.bind(("", port))
        s.close()
        print(port)
        raise SystemExit(0)
    except OSError:
        try:
            s.close()
        except Exception:
            pass
        port += 1
print(-1)
raise SystemExit(0)
PY
}

FREE_PORT="$(ensure_free_master_port "$MASTER_PORT")"
if [ "$FREE_PORT" = "-1" ]; then
    echo "ERROR: No free port found in range [$MASTER_PORT, $((MASTER_PORT + 199))]." >&2
    exit 1
fi
if [ "$FREE_PORT" != "$MASTER_PORT" ]; then
    echo "WARNING: MASTER_PORT=$MASTER_PORT is already in use, switching to MASTER_PORT=$FREE_PORT"
    export MASTER_PORT="$FREE_PORT"
fi
LOG_DIR="${LOG_DIR:-/mnt/lyaa/my_llava/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_UCIT_${UCIT_RUN_ID}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "UCIT run id: $UCIT_RUN_ID"
echo "UCIT task range: ${UCIT_START_TASK}-${UCIT_END_TASK}"
echo "UCIT task set: $UCIT_TASK_SET"
echo "UCIT train config root: $TRAIN_CONFIG_ROOT"
echo "UCIT eval config root: $EVAL_CONFIG_ROOT"
echo "UCIT only: $UCIT_ONLY"
echo "UCIT smoke tasks override: ${UCIT_SMOKE_TASKS:-<unset>}"
echo "UCIT tasks: ${UCIT_TASKS:-<range>}"
echo "UCIT eval mode: $UCIT_EVAL_MODE"
echo "UCIT reinstall between tasks: $UCIT_REINSTALL_BETWEEN_TASKS"
echo "UCIT max train samples: ${UCIT_MAX_TRAIN_SAMPLES:-<unset>}"
echo "UCIT max eval samples: ${UCIT_MAX_SAMPLES:-<unset>}"
echo "UCIT max new tokens: ${UCIT_MAX_NEW_TOKENS:-<unset>}"
echo "UCIT skip vizwiz: $UCIT_SKIP_VIZWIZ"
echo "UCIT skip flickr: $UCIT_SKIP_FLICKR"

pip_install_project() {
    pip install -e . --no-build-isolation || pip install . --no-build-isolation
}

if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip_install_project
fi

if [ "$UCIT_SMOKE" = "1" ]; then
    DATA_TASK1="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json"
    TRAIN_TASK1="$CONFIG_ROOT/train/task1_smoke.json"
    DATA_TASK2="$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json"
    TRAIN_TASK2="$CONFIG_ROOT/train/task2_smoke.json"
    DATA_TASK3="$HARD_PATH/configs/data_configs/UCIT/VizWiz.json"
    TRAIN_TASK3="$CONFIG_ROOT/train/task3_smoke.json"
    DATA_TASK4="$HARD_PATH/configs/data_configs/UCIT/IconQA.json"
    TRAIN_TASK4="$CONFIG_ROOT/train/task4_smoke.json"
    DATA_TASK5="$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json"
    TRAIN_TASK5="$CONFIG_ROOT/train/task5_smoke.json"
    DATA_TASK6="$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json"
    TRAIN_TASK6="$CONFIG_ROOT/train/task6_smoke.json"
else
    DATA_TASK1="$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json"
    TRAIN_TASK1="$CONFIG_ROOT/train/task1.json"
    DATA_TASK2="$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json"
    TRAIN_TASK2="$CONFIG_ROOT/train/task2.json"
    DATA_TASK3="$HARD_PATH/configs/data_configs/UCIT/VizWiz.json"
    TRAIN_TASK3="$CONFIG_ROOT/train/task3.json"
    DATA_TASK4="$HARD_PATH/configs/data_configs/UCIT/IconQA.json"
    TRAIN_TASK4="$CONFIG_ROOT/train/task4.json"
    DATA_TASK5="$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json"
    TRAIN_TASK5="$CONFIG_ROOT/train/task5.json"
    DATA_TASK6="$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json"
    TRAIN_TASK6="$CONFIG_ROOT/train/task6.json"
fi

task_selected() {
    local task_id="$1"
    if [ "$task_id" = "3" ] && [ "$UCIT_SKIP_VIZWIZ" = "1" ]; then
        return 1
    fi
    if [ "$task_id" = "6" ] && [ "$UCIT_SKIP_FLICKR" = "1" ]; then
        return 1
    fi
    if [ -n "$UCIT_TASKS" ]; then
        case ",$UCIT_TASKS," in
            *",$task_id,"*) return 0 ;;
            *) return 1 ;;
        esac
    fi
    [ "$UCIT_START_TASK" -le "$task_id" ] && [ "$UCIT_END_TASK" -ge "$task_id" ]
}

maybe_reinstall() {
    if [ "$UCIT_REINSTALL_BETWEEN_TASKS" = "1" ] && [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
        pip_install_project
    fi
}

LAST_EVAL_TASK=""

if task_selected 1; then
    if [ "$UCIT_ONLY" != "eval" ]; then
        bash scripts/MCITlib/Train/Task1.sh \
            $HARD_PATH/configs/modal_configs/llava.json \
            $DATA_TASK1 \
            $TRAIN_TASK1
    fi
    if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "per_task" ]; then
        CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1
    fi
    LAST_EVAL_TASK="1"
fi

if task_selected 2; then
    if [ "$UCIT_ONLY" != "eval" ]; then
        maybe_reinstall
        bash scripts/MCITlib/Train/Taskn.sh \
            $HARD_PATH/configs/modal_configs/llava.json \
            $DATA_TASK2 \
            $TRAIN_TASK2
    fi
    if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "per_task" ]; then
        CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2
    fi
    LAST_EVAL_TASK="2"
fi

if task_selected 3; then
    if [ "$UCIT_ONLY" != "eval" ]; then
        maybe_reinstall
        bash scripts/MCITlib/Train/Taskn.sh \
            $HARD_PATH/configs/modal_configs/llava.json \
            $DATA_TASK3 \
            $TRAIN_TASK3
    fi
    if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "per_task" ]; then
        CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3
    fi
    LAST_EVAL_TASK="3"
fi

if task_selected 4; then
    if [ "$UCIT_ONLY" != "eval" ]; then
        maybe_reinstall
        bash scripts/MCITlib/Train/Taskn.sh \
            $HARD_PATH/configs/modal_configs/llava.json \
            $DATA_TASK4 \
            $TRAIN_TASK4
    fi
    if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "per_task" ]; then
        CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4
    fi
    LAST_EVAL_TASK="4"
fi

if task_selected 5; then
    if [ "$UCIT_ONLY" != "eval" ]; then
        maybe_reinstall
        bash scripts/MCITlib/Train/Taskn.sh \
            $HARD_PATH/configs/modal_configs/llava.json \
            $DATA_TASK5 \
            $TRAIN_TASK5
    fi
    if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "per_task" ]; then
        CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5
    fi
    LAST_EVAL_TASK="5"
fi

if task_selected 6; then
    if [ "$UCIT_ONLY" != "eval" ]; then
        maybe_reinstall
        bash scripts/MCITlib/Train/Taskn.sh \
            $HARD_PATH/configs/modal_configs/llava.json \
            $DATA_TASK6 \
            $TRAIN_TASK6
    fi
    if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "per_task" ]; then
        CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6
    fi
    LAST_EVAL_TASK="6"
fi

if [ "$UCIT_ONLY" != "train" ] && [ "$UCIT_EVAL_MODE" = "final" ] && [ -n "$LAST_EVAL_TASK" ]; then
    CONFIG_ROOT="$EVAL_CONFIG_ROOT" bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh "$LAST_EVAL_TASK"
fi
