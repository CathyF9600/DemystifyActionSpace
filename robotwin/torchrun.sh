#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.bandw.top}
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR:${PYTHONPATH:-}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-True}

name=${name:-rel_ee_cnt_rot}
port=${port:-18001}
output_dir=${output_dir:-$REPO_ROOT/runnings/robotwin/rel_ee_cnt_rot_new}
train_metas_path=${train_metas_path:-/home/fyc/EmpiricalStudyForVLA/datasets/meta_files/rel_ee_single_camera-50-10.jsonl}
pt_path=${pt_path:-encoded_language.pt}
iters=${iters:-80000}
save_interval=${save_interval:-20000}
batch_size=${batch_size:-32}
precision=${precision:-no}
learning_rate=${learning_rate:-5e-4}
weight_decay=${weight_decay:-0}
nproc_per_node=${NPROC_PER_NODE:-1}

mkdir -p "$output_dir"
cp "$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")" "$output_dir/run.sh"

if [ -f /home/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/anaconda3/etc/profile.d/conda.sh
    conda deactivate || true
    conda activate em || true
fi

${PYTHON:-python3} -m torch.distributed.run --nproc-per-node="$nproc_per_node" --nnodes=1 --node-rank=0 --master-addr=localhost --master-port="$port" train.py \
    --model "$name" \
    --wandb_name "$name" \
    --delta_type chunk \
    --rot_repr rot6d \
    --iters "$iters" \
    --save_interval "$save_interval" \
    --batch-size "$batch_size" \
    --precision "$precision" \
    --learning_rate "$learning_rate" \
    --output_dir "$output_dir" \
    --train_metas_path "$train_metas_path" \
    --pt_path "$pt_path" \
    --port "$port" \
    --weight_decay "$weight_decay"

