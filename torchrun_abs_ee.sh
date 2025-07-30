export CUDA_VISIBLE_DEVICES=0,1
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=True

port=13555
# source /home/anaconda3/etc/profile.d/conda.sh
source /home/dodo/miniconda3/etc/profile.d/conda.sh

conda deactivate
conda activate em
# pip install fastapi
torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
  --model_type flow-matching \
  --batch-size 32 \
  --dim_actions 20 \
  --learning_rate 5e-4 \
  --precision bf16 \
  --port $port \
  --output_dir runnings/RoboTwin/abs_ee \
  --metas_path /home/fyc/EmpiricalStudyForVLA/datasets/meta_files/robotwin2_abs_ee.jsonl  
