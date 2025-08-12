#!/bin/bash
port=8019
export CUDA_VISIBLE_DEVICES=1 #4,5,6,7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com


ckpt_path=/data/empirical/aug7/abs_ee_cnt_t_no_proprio_adjust/ckpt-20000
# ckpt_path=/data/empirical/aug5/abs_ee_cnt_mlp_no_proprio/ckpt-400000
source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
env > script_env.txt
python /home/dodo/fyc/EmpiricalStudyForVLA/deploy.py \
    --ckpt_path $ckpt_path \
    --decoder_name transformer_decoder_base \
    --dim_actions 20 \
    --dim_proprio 20 \
    --num_actions 10 \
    --model_type continuous \
    --host 0.0.0.0 \
    --meta_files /home/dodo/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_ee.jsonl \
    --port $port #&





