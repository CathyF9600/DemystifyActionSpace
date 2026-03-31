export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_BASE_URL=https://api.bandw.top
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=True



name=rel_ee_cnt_rot
port=18001
output_dir=runnings/robotwin/rel_ee_cnt_rot_new

mkdir -p $output_dir
cp $0 $output_dir/run.sh

source /home/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate em
# pip install fastapi
torchrun --nproc-per-node=8 --nnodes=1 --node-rank=0 --master-addr=localhost --master-port=$port train.py \
    --model $name \
    --wandb_name $name \
    --delta_type chunk \
    --rot_repr rot6d \
    --iters 80000 \
    --save_interval 20000 \
    --batch-size 32 \
    --precision no \
    --learning_rate 5e-4 \
    --output_dir $output_dir \
    --train_metas_path /home/fyc/EmpiricalStudyForVLA/datasets/meta_files/rel_ee_single_camera-50-10.jsonl \
    --pt_path encoded_language.pt \
    --port $port \
    --weight_decay 0

