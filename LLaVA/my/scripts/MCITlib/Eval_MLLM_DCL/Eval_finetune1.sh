# #!/bin/bash

TASK_ID=$1
HARD_PATH=/mnt/lyaa/MCITlib
CONFIG_ROOT=${CONFIG_ROOT:-$HARD_PATH/configs/train_configs/MyMethod/LLaVA/MLLM-DCL}

if [ "$TASK_ID" == "1" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $CONFIG_ROOT/eval/task1.json
elif [ "$TASK_ID" == "2" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $CONFIG_ROOT/eval/task2.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $CONFIG_ROOT/eval/task2.json
elif [ "$TASK_ID" == "3" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $CONFIG_ROOT/eval/task3.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $CONFIG_ROOT/eval/task3.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $CONFIG_ROOT/eval/task3.json
elif [ "$TASK_ID" == "4" ]; then
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $CONFIG_ROOT/eval/task4.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $CONFIG_ROOT/eval/task4.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $CONFIG_ROOT/eval/task4.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $CONFIG_ROOT/eval/task4.json
else
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $CONFIG_ROOT/eval/task5.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $CONFIG_ROOT/eval/task5.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $CONFIG_ROOT/eval/task5.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $CONFIG_ROOT/eval/task5.json
    pip install -e .
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh $HARD_PATH/configs/modal_configs/llava.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $CONFIG_ROOT/eval/task5.json
fi
