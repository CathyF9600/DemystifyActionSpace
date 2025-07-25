export CUDA_VISIBLE_DEVICE=7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=True
gpu=7
port=13555

torchrun --nproc-per-node=2 --nnodes=1 --node-rank=$gpu --master-addr=localhost --master-port=$port train.py \
  --model_type flow-matching \
  --dim_actions 16 \
  --batch-size 32 \
  --learning_rate 5e-4 \
  --precision fp16 \
  --port $port \
  --output_dir runnings/RoboTwin/abs_qpos \
  --metas_path /home/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_qpos.jsonl  
