export CUDA_VISIBLE_DEVICES=1,2,3,4
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=True

name=abs_ee_cnt_mean
port=13535
source /home/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
# pip install fastapi
torchrun --nproc-per-node=4 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
    --model $name \
    --wandb_name $name \
    --iters 1000000 \
    --save_interval 20000 \
    --batch-size 64 \
    --precision no \
    --learning_rate 1e-4 \
    --output_dir runnings/RoboTwin/$name \
    --train_metas_path datasets/meta_files/abs_ee_single_camera-50.jsonl \
    --pt_path encoded_language.pt \
    --save_interval 10000 \
    --port 29530 \
    --weight_decay 0
