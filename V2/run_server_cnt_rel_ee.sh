#!/bin/bash
port=18867
export CUDA_VISIBLE_DEVICES=0 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path='/home/dodo/fyc/zhengjl-ckpt/all'
# ckpt_path=/home/dodo/fyc/EmpiricalStudyForVLA/V2/exp_dodo/cnt-100-10-mlp6/rel_ee/ckpt-final
# stats_path=/home/dodo/fyc/EmpiricalStudyForVLA/datasets_dodo/meta_files
# eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/cnt-100-10-mlp6/rel_ee
# model_name='model_rel_ee_cnt_mlp6'

ckpt_path=/home/dodo/fyc/EmpiricalStudyForVLA/V2/exp_dodo/cnt-200/rel_ee/ckpt-final
stats_path=/home/dodo/fyc/EmpiricalStudyForVLA/datasets_dodo/meta_files/200data10task
eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/cnt-200/rel_ee_retrain
model_name='model_rel_ee_cnt'

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
python deploy.py \
    --ckpt_path $ckpt_path \
    --stats_path $stats_path \
    --norm_action True \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port &

sleep 30

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate RoboTwin
# pip install json-numpy
# pip install uvicorn
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
    --task_config demo_randomized
    # --hdf5_path /home/dodo/fyc/RoboTwin/data/adjust_bottle/demo_randomized/data/episode0.hdf5
    #> $eval_log_dir/log.txt 2>&1

PID=$(lsof -i :$port -t)
kill -9 $PID
