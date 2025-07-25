# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import os
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
import subprocess
from accelerate import Accelerator
from dataset import create_dataloader
from model import BaseModel, language_encoder
from timm import create_model
from safetensors.torch import load_file
from accelerate.utils import DistributedDataParallelKwargs

def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    # Base Settings
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--iters', default=1000000, type=int)
    parser.add_argument('--metas_path', default='', type=str)
    parser.add_argument('--precision', default='fp16', type=str)
    
    # Model parameters
    parser.add_argument('--vision_backbone', default="resnet18.a1_in1k", type=str, help="Vision backbone name (from timm)")
    parser.add_argument('--decoder_name', default="mlp_decoder_base", type=str, help="Decoder name")
    parser.add_argument('--model_type', type=str, default="continuous", choices=["continuous", "discrete", "flow-matching"], help="Model type")
    parser.add_argument('--num_actions', type=int, default=10, help="Number of action chunks")
    parser.add_argument('--learning_coef', default=1., type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    
    parser.add_argument('--seed', default=0, type=int)
    
    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=20000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--dim_actions', default=20, type=int)
    
    parser.add_argument('--output_dir', default='runnings/',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--resume', default=None, help='model resume from checkpoint')
    parser.add_argument('--pretrained', default=None, help='model load pretraining param')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29531, type=int, help='port')

    return parser

def main(args):
    output_dir = Path(args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision=args.precision,
                              log_with="tensorboard", 
                              project_dir=output_dir, kwargs_handlers=[kwargs])
    accelerator.init_trackers("HDP_Training", config=vars(args))
    torch.distributed.barrier()
    
    # Initialize language encoder (frozen)
    lang_encoder = language_encoder()
    lang_encoder.to(accelerator.device)
    lang_encoder.eval()  # No training for language encoder
    
    # Initialize policy model
    model = BaseModel(
        vision_backbone=args.vision_backbone,
        model_type=args.model_type,
        decoder_name=args.decoder_name,
        num_action_chunk=args.num_actions,
        dim_actions=args.dim_actions  # Matches dataset's action dimension
    )
    model.to(accelerator.device)
    
    if args.pretrained is not None:
        accelerator.print('>>>>>> load pretrain from {}'.format(args.pretrained))
        print(model.load_state_dict(load_file(args.pretrained), strict=False))
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    accelerator.print(f'number of params: {n_parameters} M')

    train_dataloader = create_dataloader(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        batch_size=args.batch_size,
        metas_path=args.metas_path,
        num_actions=args.num_actions+1
    )
    
    model = model.to(torch.float32)
    
    # 设置优化器参数组
    encoder_params = list(map(id, model.vision_backbone.parameters())) 
    other_params = filter(lambda p: id(p) not in encoder_params, model.parameters()) 

    # optim = torch.optim.AdamW([
    #         {'params': model.vision_backbone.parameters(), 
    #             'lr': args.learning_rate * args.learning_coef,
    #             'weight_decay': args.weight_decay * args.learning_coef},
            
    #         {'params': other_params,  
    #             'lr': args.learning_rate,
    #             'weight_decay': args.weight_decay}
    #     ],
    #     betas=(0.9, 0.95)
    # )
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    model, optim = accelerator.prepare(model, optim)
    if args.resume is not None:
        accelerator.print('>>>>>> resume from {}'.format(args.resume))
        accelerator.load_state(args.resume)
    
    model.train()
    train_dataloader = iter(train_dataloader)
    
    accelerator.print(f"Start training for {args.iters} iters")
    for iters in range(args.iters):
        past_time = time.time()
        data = next(train_dataloader)
        torch.distributed.barrier()
        # language_instruction = text_processor.encode_language(data['language_instruction'])
        # print('*******', data.keys())
        language_instruction = lang_encoder(data['language_instruction']) 
        del data['language_instruction']
        # inputs = {
        #     **{key: value.cuda(non_blocking=True) for key, value in data.items()},
        #     **{key: value.cuda(non_blocking=True) for key, value in language_instruction.items()}
        # }
        inputs = {
            **{key: value.cuda(non_blocking=True) for key, value in data.items()},
            "encoded_language": language_instruction.cuda(non_blocking=True)
        }
        optim.zero_grad()
        # print('inputs', inputs.keys())
        loss = model(**inputs)
        # loss = sum(loss_dict.values())
        accelerator.backward(loss)
        optim.step()
        #### log
        # loss_dict = {key: value.item() for key, value in loss_dict.items()}
        if iters % args.log_interval == 0: 
            # accelerator.log(loss_dict, step=iters)
            accelerator.log({'loss': loss.item()}, step=iters)
            accelerator.print(f"[Iter {iters}] [Training Loss] {loss.item()} [time_per_iter] {time.time() - past_time}")
        if iters % args.save_interval == 0 and iters != 0:
            accelerator.print("========start saving models=========")
            accelerator.save_state(os.path.join(output_dir, f"ckpt-{iters}"))
        torch.distributed.barrier()
    accelerator.save_state(os.path.join(output_dir, f"ckpt-final"))


def slurm_env_init(args):
    args.rank = int(os.environ.get('RANK', 0))
    args.world_size = int(os.environ['WORLD_SIZE'])
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)
    args.gpu = args.rank
    torch.cuda.set_device(args.gpu)
    
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    
    torch.distributed.init_process_group(backend="nccl", rank=args.rank, world_size=args.world_size)
    print('ddp end init')
    torch.distributed.barrier()
    
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("ddp setup done")
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(slurm_env_init(args))