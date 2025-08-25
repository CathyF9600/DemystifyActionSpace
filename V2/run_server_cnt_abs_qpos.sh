#!/bin/bash
port=18892
export CUDA_VISIBLE_DEVICES=1 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path='/home/dodo/fyc/zhengjl-ckpt/all'
ckpt_path=/data/empirical/cnt-50/abs_qpos_20t
# /data/empirical/aug12/cnt_abs_qpos/ckpt-25w
model_name='model_abs_qpos_cnt'
stats_path=/data/empirical/cnt-50/abs_qpos_20t

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
eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/cnt-50/abs_qpos_20t
cd /home/dodo/fyc/RoboTwin
python script/robotwin_client_v2.py \
    --data_type abs \
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
    --instruction_type seen #> $eval_log_dir/log.txt 2>&1

PID=$(lsof -i :$port -t)
kill -9 $PID