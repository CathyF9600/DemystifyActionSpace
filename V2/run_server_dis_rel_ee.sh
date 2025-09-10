#!/bin/bash
port=18891
export CUDA_VISIBLE_DEVICES=1 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path='/home/dodo/fyc/zhengjl-ckpt/all'
ckpt_path=/data/empirical/dis-50/rel_ee_retrain
stats_path=/data/empirical/dis-50
model_name='model_rel_ee_dis'

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
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
eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/dis-50/rel_ee_retrain
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
    --instruction_type seen

PID=$(lsof -i :$port -t)
kill -9 $PID