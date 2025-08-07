srun -p mozi_t  -n8  --gres=gpu:8  --ntasks-per-node=8  \
    python -u slurm_train.py \
        --model model_base \
        --iters 1000000 \
        --batch-size 64 \
        --precision no \
        --learning_rate 1e-4 \
        --output_dir /mnt/petrelfs/zhengjinliang/EmpiricalStudyForVLA/V2/exp \
        --train_metas_path /mnt/petrelfs/zhengjinliang/EmpiricalStudyForVLA/V2/robotwin-100.json \
        --save_interval 10000 \
        --port 29530 \
        --weight_decay 0