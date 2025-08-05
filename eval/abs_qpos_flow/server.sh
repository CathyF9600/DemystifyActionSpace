#!/bin/bash
port=8011
export CUDA_VISIBLE_DEVICES=1 #4,5,6,7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
# # ckpt_path=/data/empirical/abs_ee
# # ckpt_path=/data/empirical/abs_ee/ckpt-400000
# ckpt_path=/data/empirical/abs_ee_adjust/ckpt-20000
# # Correct path to conda.sh based on your Conda installation
# # source /home/anaconda3/etc/profile.d/conda.sh

# source /home/dodo/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate em
# python /home/dodo/fyc/EmpiricalStudyForVLA/deploy.py \
#     --ckpt_path $ckpt_path \
#     --model_type flow-matching \
#     --host 0.0.0.0 \
#     --meta_files /home/dodo/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_ee.jsonl \
#     --port $port #&





# ckpt_path=/data/empirical/abs_qpos_flow/ckpt-final
ckpt_path=/data/empirical/abs_qpos_flow/mlp6/ckpt-220000
source /home/dodo/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
python /home/dodo/fyc/EmpiricalStudyForVLA/deploy.py \
    --ckpt_path $ckpt_path \
    --decoder_name mlp_decoder_large \
    --dim_actions 14 \
    --dim_proprio 14 \
    --num_actions 30 \
    --model_type flow-matching \
    --host 0.0.0.0 \
    --meta_files /home/dodo/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_qpos.jsonl \
    --port $port #&













# sleep 30

# conda deactivate
# conda activate RoboTwin
# # pip install json-numpy
# # pip install uvicorn
# eval_log_dir=/home/dodo/fyc/HeteroDiffusionPolicy/AbsEEFFlowV4/runnings/RoboTwin/ckpt-40000/eval
# cd /home/dodo/fyc/RoboTwin
# python script/robotwin_client.py \
#     --host 0.0.0.0 \
#     --port $port \
#     --eval_log_dir $eval_log_dir \
#     --num_episodes 10 \
#     --device 0 \
#     --task_name open_microwave \
#     --output_path $eval_log_dir \
#     --task_config demo_randomized #> $eval_log_dir/log.txt

# PID=$(lsof -i :$port -t)
# kill -9 $PID