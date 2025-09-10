#!/bin/bash
port=18889
export CUDA_VISIBLE_DEVICES=0 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path='/home/dodo/fyc/zhengjl-ckpt/all'
# ckpt_path=/data/empirical/aug12/cnt_abs_ee/ckpt-75w
ckpt_path=/home/fyc/EmpiricalStudyForVLA/V2/exp_A100/cnt-50/abs_ee/ckpt-final # 50 data 10 task
model_name='model_abs_ee_cnt'

stats_path=/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/abs_ee_single_camera-50-10_global_stats_rel.npz

# source /home/dodo/miniconda3/etc/profile.d/conda.sh
source /home/anaconda3/etc/profile.d/conda.sh #A100

conda deactivate
conda activate em
python deploy_loss.py \
    --ckpt_path $ckpt_path \
    --stats_path $stats_path \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port &


sleep 30

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
    --port $port \
    --ctrl_interface abs_ee

PID=$(lsof -i :$port -t)
kill -9 $PID