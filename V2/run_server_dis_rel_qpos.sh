#!/bin/bash
port=18892
export CUDA_VISIBLE_DEVICES=1 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
ckpt_path=/data/empirical/dis/rel-qpos-50w
# ckpt_path=/data/empirical/dis-100/rel_qpos
stats_path=/data/empirical/dis
# ckpt_path=/data/empirical/dis-25/rel_qpos
# stats_path=/data/empirical/dis-25
model_name='model_rel_qpos_dis'

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
# env > script_env_bug.txt
python deploy.py \
    --ckpt_path $ckpt_path \
    --stats_path $stats_path \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port &

sleep 30

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate RoboTwin
# pip install json-numpy
# pip install uvicorn
# eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/dis-100/rel_qpos
eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/dis/rel_qpos_50w_retrain
cd /home/dodo/fyc/RoboTwin
python script/robotwin_client_v2.py \
    --data_type rel \
    --action_type qpos \
    --host 0.0.0.0 \
    --port $port \
    --eval_log_dir $eval_log_dir \
    --num_episodes 10 \
    --device 0 \
    --seed 3 \
    --task_name all \
    --output_path $eval_log_dir \
    --task_config demo_randomized \
    --instruction_type seen

PID=$(lsof -i :$port -t)
kill -9 $PID