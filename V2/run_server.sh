#!/bin/bash
port=18885
export CUDA_VISIBLE_DEVICES=0 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path='/home/dodo/fyc/zhengjl-ckpt/all'
ckpt_path=/data/empirical/aug10/cnt_abs_ee/ckpt-25w
model_name='model_abs_ee_cnt'

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
env > script_env_bug.txt
python deploy.py \
    --ckpt_path $ckpt_path \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port
