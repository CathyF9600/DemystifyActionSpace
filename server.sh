#!/bin/bash
port=8000
export CUDA_VISIBLE_DEVICES=0 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
source /home/agilex/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate das

ckpt_path=real-world/exp_bowl/abs_joint/ckpt-final
model_name='model_abs_joint_act'
python deploy.py \
    --ckpt_path $ckpt_path \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port

