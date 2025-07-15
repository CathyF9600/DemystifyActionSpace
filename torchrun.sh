export CUDA_VISIBLE_DEVICE=0,1
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
port=13553


torchrun --nproc-per-node=2 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
  --model HFP_large \
  --batch-size 1 \
  --learning_rate 1e-4 \
  --precision fp16 \
  --port $port \
  --output_dir runnings/RoboTwin \
  --metas_path /home/dodo/fyc/HeteroDiffusionPolicy/HeteroFlowPolicy/datasets/meta_files/robotwin.jsonl