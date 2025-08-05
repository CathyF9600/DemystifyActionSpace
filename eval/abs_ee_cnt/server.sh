#!/bin/bash
port=8019
export CUDA_VISIBLE_DEVICES=1 #4,5,6,7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com


ckpt_path=/data/empirical/abs_ee_cnt/ckpt-final
source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
python /home/dodo/fyc/EmpiricalStudyForVLA/deploy.py \
    --ckpt_path $ckpt_path \
    --dim_actions 20 \
    --dim_proprio 20 \
    --num_actions 30 \
    --model_type continuous \
    --host 0.0.0.0 \
    --meta_files /home/dodo/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_ee.jsonl \
    --port $port #&





