#!/bin/bash
port=8008
export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com

# Correct path to conda.sh based on your Conda installation
# source /home/anaconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate hetero
# # pip install fastapi
# python /home/fyc/HeteroDiffusionPolicy/AbsEEFFlowV4/deploy.py \
#     --ckpt_path /home/fyc/HeteroDiffusionPolicy/AbsEEFFlowV4/runnings/RoboTwin/ckpt-40000 \
#     --model_name HFP_large \
#     --host 0.0.0.0 \
#     --meta_files /home/fyc/HeteroDiffusionPolicy/HeteroFlowPolicy/datasets/meta_files/robotwin.jsonl \
#     --port $port &

# sleep 30
source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
# pip install json-numpy
# pip install uvicorn
eval_log_dir=/home/fyc/EmpiricalStudyForVLA/eval/abs_ee/log
cd /home/fyc/RoboTwin
python script/robotwin_client.py \
    --host 0.0.0.0 \
    --port $port \
    --eval_log_dir $eval_log_dir \
    --num_episodes 100 \
    --device 0 \
    --seed 3 \
    --task_name adjust_bottle \
    --output_path $eval_log_dir \
    --task_config demo_randomized \
    --instruction_type seen > $eval_log_dir/log.txt 2>&1

# PID=$(lsof -i :$port -t)
# kill -9 $PID