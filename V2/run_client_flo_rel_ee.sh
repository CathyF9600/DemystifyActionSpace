#!/bin/bash
port=18875
export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate RoboTwin
# pip install json-numpy
# pip install uvicorn
eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/flow_rel_ee-75w
cd /home/dodo/fyc/RoboTwin
python script/robotwin_client_v2.py \
    --data_type rel \
    --action_type ee \
    --host 0.0.0.0 \
    --port $port \
    --eval_log_dir $eval_log_dir \
    --num_episodes 10 \
    --device 0 \
    --seed 3 \
    --task_name all \
    --output_path $eval_log_dir \
    --task_config demo_randomized \
    --instruction_type seen #> $eval_log_dir/log.txt 2>&1

# PID=$(lsof -i :$port -t)
# kill -9 $PID