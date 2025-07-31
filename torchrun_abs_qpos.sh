export CUDA_VISIBLE_DEVICES=6,7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=True
export WANDB_API_KEY=56c323ace61a5076f5d8e92a91237607bbc362a7

port=13556
source /home/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
# pip install wandb
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
  --model_type flow-matching \
  --dim_actions 14 \
  --dim_proprio 14 \
  --batch-size 32 \
  --learning_rate 5e-4 \
  --precision bf16 \
  --port $port \
  --output_dir runnings/RoboTwin/abs_qpos_flow \
  --wandb_name robotwin2_abs_qpos_flow \
  --metas_path /home/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_qpos.jsonl  
