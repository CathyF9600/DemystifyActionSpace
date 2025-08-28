#!/bin/bash
port=8017
export CUDA_VISIBLE_DEVICES=0 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path=/data/empirical/aug12/cnt_abs_ee/ckpt-75w
ckpt_path=/home/agilex/empirical/abs_ee_aug27_pretrained
# model_name='model_abs_ee_cnt'
model_name='model_abs_ee_cnt30'

stats_path=/home/agilex/empirical/abs_ee_aug26_act30

 
# conda deactivate
# conda activate em
python deploy.py \
    --ckpt_path $ckpt_path \
    --stats_path $stats_path \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port &

