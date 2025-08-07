ckpt_path='/data/HDP/HFP/20250704/40K'
model_name='model_base'
port=18882
python deploy.py \
    --ckpt_path $ckpt_path \
    --model_name $model_name \
    --host 0.0.0.0 \
    --port $port
