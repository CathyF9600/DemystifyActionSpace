#!/bin/bash
port=18885
export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com

source /home/anaconda3/etc/profile.d/conda.sh #A100
conda deactivate
conda activate RoboTwin
# pip install json-numpy
# pip install uvicorn
# eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/cnt_abs_ee-25w
python plot.py \
    --host 0.0.0.0 \
    --port $port