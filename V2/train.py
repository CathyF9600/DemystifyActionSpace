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
import model
from timm import create_model
from safetensors.torch import load_file
from accelerate.utils import DistributedDataParallelKwargs
import torch.nn.functional as F
import wandb
from mmengine import fileio
import io
import h5py
import json
from scipy.spatial.transform import Rotation as R

def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    # Base Settings
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--iters', default=1000000, type=int)
    parser.add_argument('--train_metas_path', type=str)
    parser.add_argument('--precision', default='no', type=str)
    
    parser.add_argument('--model', default='model_base', type=str)
    parser.add_argument('--model_type', type=str, default="continuous", choices=["continuous", "discrete", "flow-matching"], help="Model type")
    parser.add_argument('--num_bins', default=256, type=int)

    parser.add_argument('--learning_coef', default=1., type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--seed', default=0, type=int)
    
    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=20000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    
    
    parser.add_argument('--output_dir', default='runnings/',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--resume', default=None, help='model resume from checkpoint')
    parser.add_argument('--pretrained', default=None, help='model load pretraining param')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29531, type=int, help='port')

    parser.add_argument('--wandb_name', default='robotwin2_abs_qpos', type=str)

    return parser

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

# def cal_delta_rotate(q1, q2):
#     q1 = R.from_quat(q1)
#     q2 = R.from_quat(q2)
#     del_rotate = q1 * q2.inv()
#     return del_rotate.as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def cal_delta_rotate(q1, q2):
    q1 = R.from_quat(q1)
    q2 = R.from_quat(q2)
    del_rotate = q1 * q2.inv()
    return del_rotate.as_matrix()[..., :, :2].reshape(q1.as_quat().shape[:-1] + (6,))

def compute_mean_std(hdf5_paths, control='ee'):
    all_data = []
    for path in hdf5_paths:
        with h5py.File(path, 'r') as data:
            if control == 'qpos':
                left_joint = data["joint_action/left_arm"][()]      # (T, 7)
                right_joint = data["joint_action/right_arm"][()]    # (T, 7)
                left_grip = data["joint_action/left_gripper"][()]   # (T,)
                right_grip = data["joint_action/right_gripper"][()] # (T,)
                left_grip = 1 - left_grip * 2
                right_grip = 1 - right_grip * 2

                joint_diff = np.concatenate([
                    left_joint[1:] - left_joint[:-1],
                    left_grip[1:, None],
                    right_joint[1:] - right_joint[:-1],
                    right_grip[1:, None]
                ], axis=-1)
                action_seq = joint_diff

            else:
                left_pos = data["endpose/left_endpose"][()]
                right_pos = data["endpose/right_endpose"][()]
                left_grip = data["endpose/left_gripper"][()]
                right_grip = data["endpose/right_gripper"][()]
                left_grip = 1 - left_grip * 2
                right_grip = 1 - right_grip * 2

                left_delta_xyz = left_pos[1:, :3] - left_pos[:-1, :3]
                right_delta_xyz = right_pos[1:, :3] - right_pos[:-1, :3]

                left_delta_rot6d = cal_delta_rotate(left_pos[1:, 3:], left_pos[:-1, 3:])
                right_delta_rot6d = cal_delta_rotate(right_pos[1:, 3:], right_pos[:-1, 3:])

                action_seq = np.concatenate([
                    left_delta_xyz,
                    left_delta_rot6d,
                    left_grip[1:, None],
                    right_delta_xyz,
                    right_delta_rot6d,
                    right_grip[1:, None]
                ], axis=-1)

            all_data.append(action_seq)

    stacked = np.concatenate(all_data, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    min_val = stacked.min(axis=0)
    max_val = stacked.max(axis=0)

    print('mean:', mean)
    print('std:', std)
    print('min:', min_val)
    print('max:', max_val)

    return mean, std, min_val, max_val


def get_hdf5s(metas_path):
    metas = {}

    # reading setting
    if fileio.isdir(metas_path): meta_files = fileio.list_dir_or_file(metas_path, suffix='.json', recursive=True, list_dir=False)
    else: meta_files, metas_path = [metas_path], ""
    for file in meta_files:
        with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
            meta = json.load(f)
            print(f"================detect dataset {meta['dataset_name']} with traj {len(meta['datalist'])}==================")
            random.shuffle(meta['datalist'])
            metas[meta['dataset_name']] = meta
    return meta['datalist']

def main(args):
    output_dir = Path(args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision = args.precision,
                              log_with="tensorboard", 
                              project_dir=output_dir, kwargs_handlers=[kwargs])
    accelerator.init_trackers("HFP_Training")
    torch.distributed.barrier()
    model, _ = create_model(args.model)
    if args.pretrained is not None:
        accelerator.print('>>>>>> load pretrain from {}'.format(args.pretrained))
        print(model.load_state_dict(load_file(args.pretrained), strict=False))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    accelerator.print(f'number of params: {n_parameters} M')

    if True: #rank == 0 and 'rel' in args.wandb_name:
        # paths = glob.glob(hdf5_files)
        control = None
        if 'ee' in args.wandb_name:
            control = 'ee'
        else:
            control = 'qpos'
        if 'rel' in args.wandb_name or args.model_type == 'discrete':
            hdf5_files = get_hdf5s(args.train_metas_path)
            print('len(hdf5_files)', len(hdf5_files))
            stats_file = args.train_metas_path.replace(".jsonl", "_global_stats.npz")
            print('Saving stats_file', args.train_metas_path, stats_file)
            mean, std, min_val, max_val = compute_mean_std(hdf5_files, control=control)
            np.savez(stats_file, mean=mean, std=std, min=min_val, max=max_val)
    
    train_dataloader = iter(create_dataloader(
        rank = args.rank,
        world_size = args.world_size,
        batch_size = args.batch_size,
        metas_path = args.train_metas_path,
        num_actions= model.num_action_chunk,
        model_type=args.model_type,
        num_bins=args.num_bins
    ))
    
    model = model.to(torch.float32)
    # 设置优化器参数组
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    model, optim = accelerator.prepare(model, optim)
    if args.resume is not None:
        accelerator.print('>>>>>> resume from {}'.format(args.resume))
        accelerator.load_state(args.resume)
    train_dataloader = iter(train_dataloader)
    model.train()
    accelerator.print(f"Start training for {args.iters} iters")
    for iters in range(args.iters):
        past_time = time.time()
        data = next(train_dataloader)
        inputs = {
            **{key: value.cuda(non_blocking=True) for key, value in data.items()},
        }
        optim.zero_grad()
        loss = model(**inputs)
        accelerator.backward(loss)
        optim.step()
        if iters % args.log_interval == 0: 
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                wandb.log({
                    "loss": loss.item(),
                    # **{f"log/{k}": v for k, v in log_dict.items()}
                }, step=iters)
            accelerator.log({'loss': loss.item()}, step=iters)
            accelerator.print(f"[Iter {iters}] [Training Loss] {loss.item()} [time_per_iter] {time.time() - past_time}")
            
        if iters % args.save_interval == 0 and iters != 0:
            model.eval()
            accelerator.print("========start saving models=========")
            accelerator.save_state(os.path.join(output_dir, f"last_checkpoint"),
                                   safe_serialization=True)
            model.train()
        accelerator.wait_for_everyone()
    accelerator.save_state(os.path.join(output_dir, f"ckpt-final"))

# def slurm_env_init(args):
#     args.rank = int(os.environ['SLURM_PROCID'])
#     args.gpu = args.rank % torch.cuda.device_count()
#     args.world_size = int(os.environ['SLURM_NTASKS'])
#     node_list = os.environ['SLURM_NODELIST']
#     num_gpus = torch.cuda.device_count()
#     addr = subprocess.getoutput(
#         f'scontrol show hostname {node_list} | head -n1')
#     os.environ['MASTER_PORT'] = str(getattr(args, 'port', '29529'))
#     os.environ['MASTER_ADDR'] = addr
#     os.environ['WORLD_SIZE'] = str(args.world_size)
#     os.environ['LOCAL_RANK'] = str(args.rank % num_gpus)
#     os.environ['RANK'] = str(args.rank)
#     torch.cuda.set_device(args.gpu)
    
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}, gpu {}'.format(
#         args.rank, args.dist_url, args.gpu), flush=True)
    
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
    
#     # fix the seed for reproducibility
#     seed = args.seed + torch.distributed.get_rank()
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True
#     return args

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
    import os
    rank = int(os.environ.get("RANK", 0))
    print('rank', rank)
    if rank == 0:
        wandb.init(
            project="EmpiricalStudyForVLA",
            name=args.wandb_name,
            config=vars(args)
        )
    main(slurm_env_init(args))