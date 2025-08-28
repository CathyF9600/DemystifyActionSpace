#!/bin/bash
port=18892
export CUDA_VISIBLE_DEVICES=0 #4,5,6,7
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# ckpt_path='/home/dodo/fyc/zhengjl-ckpt/all'
<<<<<<< HEAD
ckpt_path=/home/fyc/EmpiricalStudyForVLA/V2/exp_A100/cnt/abs_ee/ckpt-final
model_name='model_abs_ee_cnt'

# source /home/dodo/miniconda3/etc/profile.d/conda.sh
source /home/anaconda3/etc/profile.d/conda.sh #A100
conda deactivate
conda activate em

=======
# ckpt_path=/data/empirical/aug12/cnt_abs_ee/ckpt-75w
ckpt_path=/data/empirical/cnt-50/abs_ee_40t
stats_path=/data/empirical/cnt-50/abs_ee_40t
model_name='model_abs_ee_cnt'

source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
>>>>>>> 10303c44ec3cfa69e91e86435de2df11703494e4
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
eval_log_dir=/home/dodo/fyc/EmpiricalStudyForVLA/V2/eval/cnt-50/abs_ee_40t
cd /home/dodo/fyc/RoboTwin
python script/robotwin_client_v2.py \
    --data_type abs \
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
    --instruction_type seen #> $eval_log_dir/log.txt 2>&1

PID=$(lsof -i :$port -t)
kill -9 $PID