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
from act_policy import ACTPolicy
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
    parser.add_argument('--pt_path', type=str)
    parser.add_argument('--precision', default='no', type=str)
    
    parser.add_argument('--model', default='model_base', type=str)
    parser.add_argument('--model_type', type=str, default="continuous", choices=["ACT", "continuous", "discrete", "flow-matching"], help="Model type")
    parser.add_argument('--num_bins', default=1, type=int)

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
    parser.add_argument('--normalize_action', default=False, action='store_true', help='load ckpt path')
    parser.add_argument('--normalize_proprio', default=False, action='store_true', help='load ckpt path')

    # ACT
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", required=True)
    # parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
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

def compute_mean_std(hdf5_paths, control='ee', data_type='rel', env=None):
    all_proprios = []
    all_actions = []
    print('control, data_type, env', control, data_type, env)
    for path in hdf5_paths:
        with h5py.File(path, 'r') as data:
            if data_type =='rel':
                if control == 'qpos':
                    if env == 'real':
                        proprio_seq = data['observations/qpos'][()]
                        left_joint = proprio_seq[:, :6]
                        right_joint = proprio_seq[:, 7:13]
                        left_grip = proprio_seq[:, 6]
                        right_grip = proprio_seq[:, 13]
                        joint_diff = np.concatenate([
                            left_joint[1:] - left_joint[:-1],
                            left_grip[1:, None],  # use future value directly
                            right_joint[1:] - right_joint[:-1],
                            right_grip[1:, None]
                        ], axis=-1)
                        action_seq = joint_diff
                    else:
                        left_joint = data["joint_action/left_arm"][()]      # (T, 7)
                        right_joint = data["joint_action/right_arm"][()]    # (T, 7)
                        left_grip = data["joint_action/left_gripper"][()]   # (T,)
                        right_grip = data["joint_action/right_gripper"][()] # (T,)
                        proprio_seq = np.concatenate([
                            left_joint,                        # (T,7)
                            left_grip[:, None],             # (T,1)
                            right_joint,                    # (T,7)
                            right_grip[:, None]             # (T,1)
                        ], axis=-1)
                        joint_diff = np.concatenate([
                            left_joint[1:] - left_joint[:-1],
                            left_grip[1:, None],
                            right_joint[1:] - right_joint[:-1],
                            right_grip[1:, None]
                        ], axis=-1)                
                        action_seq = joint_diff
                else: # ee
                    if env == 'real':
                        proprio_seq = data['observations/eef_quaternion'][()]
                        left_ee = proprio_seq[:, :7]
                        right_ee = proprio_seq[:, 8:15]
                        left_grip = proprio_seq[:, 7]
                        right_grip = proprio_seq[:, 15]
                        proprio_seq = np.concatenate([
                            left_ee[:, :3],
                            quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                            left_grip[:, None],             # (T,1)
                            right_ee[:, :3],
                            quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                            right_grip[:, None]             # (T,1)
                        ], axis=-1)
                        left_delta_xyz = left_ee[1:, :3] - left_ee[:-1, :3]
                        right_delta_xyz = right_ee[1:, :3] - right_ee[:-1, :3]
                        left_delta_rot6d = cal_delta_rotate(left_ee[1:, 3:], left_ee[:-1, 3:])
                        right_delta_rot6d = cal_delta_rotate(right_ee[1:, 3:], right_ee[:-1, 3:])
                        action_seq = np.concatenate([
                            left_delta_xyz,
                            left_delta_rot6d,
                            left_grip[1:, None],   # future gripper value
                            right_delta_xyz,
                            right_delta_rot6d,
                            right_grip[1:, None]
                        ], axis=-1)
                    else:
                        left_pos = data["endpose/left_endpose"][()]
                        right_pos = data["endpose/right_endpose"][()]
                        left_grip = data["endpose/left_gripper"][()]
                        right_grip = data["endpose/right_gripper"][()]
                        left_delta_xyz = left_pos[1:, :3] - left_pos[:-1, :3]
                        right_delta_xyz = right_pos[1:, :3] - right_pos[:-1, :3]
                        proprio_seq = np.concatenate([
                            left_ee[:, :3],
                            quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                            left_grip[:, None],             # (T,1)
                            right_ee[:, :3],
                            quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                            right_grip[:, None]             # (T,1)
                        ], axis=-1)
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
            if data_type == 'abs':
                if control == 'qpos':
                    if env == 'real':
                        action_seq = data['observations/qpos'][()] # 实机
                        proprio_seq = action_seq
                    else:
                        left_joint = data["joint_action/left_arm"][()]      # (T, 7)
                        right_joint = data["joint_action/right_arm"][()]    # (T, 7)
                        left_grip = data["joint_action/left_gripper"][()]   # (T,)
                        right_grip = data["joint_action/right_gripper"][()] # (T,)
                        action_seq = np.concatenate([
                            left_joint,
                            left_grip[:, None],
                            right_joint,
                            right_grip[:, None]
                        ], axis=-1)
                        proprio_seq = np.concatenate([left_joint, right_joint], axis=-1)
                else: # ee
                    if env == 'real':
                        action_seq = data['observations/eef_6d'][()] # 实机
                        proprio_seq = action_seq
                    else:
                        left_ee = data["endpose/left_endpose"][()]
                        right_ee = data["endpose/right_endpose"][()]
                        left_grip = data["endpose/left_gripper"][()]
                        right_grip = data["endpose/right_gripper"][()]
                        action_seq = np.concatenate([
                            left_ee[:, :3],
                            quat_to_rotate6D(left_ee[:, 3:]),
                            left_grip[:, None],
                            right_ee[:, :3],
                            quat_to_rotate6D(right_ee[:, 3:]),  
                            right_grip[:, None]
                        ], axis=-1)
                        proprio_seq = action_seq
            all_actions.append(action_seq)
            all_proprios.append(proprio_seq)
    # ---- Compute stats ----
    stacked_actions = np.concatenate(all_actions, axis=0)
    stacked_proprios = np.concatenate(all_proprios, axis=0)
    print('stacked_actions.shape', stacked_actions.shape, 'stacked_proprios.shape', stacked_proprios.shape)
    stats = {
        "action": {
            "mean": stacked_actions.mean(axis=0),
            "std": stacked_actions.std(axis=0),
            "min": stacked_actions.min(axis=0),
            "max": stacked_actions.max(axis=0),
            "p5": np.percentile(stacked_actions, 5, axis=0),
            "p95": np.percentile(stacked_actions, 95, axis=0),
        },
        "proprio": {
            "mean": stacked_proprios.mean(axis=0),
            "std": stacked_proprios.std(axis=0),
        }
    }
    return stats


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

def make_policy(policy_class):
    if policy_class == "ACT":
        from constants import SIM_TASK_CONFIGS
        policy_config = {
            "lr": 1e-5,
            "num_queries": 50, #args.chunk_size,
            "kl_weight": 10,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
        }

        SIM_TASK_CONFIGS["policy_config"] = policy_config
        policy = ACTPolicy(SIM_TASK_CONFIGS)
        print(">>> Create ACT Policy", SIM_TASK_CONFIGS)
    else:
        raise NotImplementedError
    return policy

def main(args):
    output_dir = Path(args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision = args.precision,
                              log_with="tensorboard", 
                              project_dir=output_dir, kwargs_handlers=[kwargs])
    accelerator.init_trackers("HFP_Training")
    torch.distributed.barrier()

    if True: #rank == 0 and 'rel' in args.wandb_name:
        # paths = glob.glob(hdf5_files)
        control = None
        if 'ee' in args.wandb_name:
            control = 'ee'
            proprio_dim = 20
        else:
            control = 'qpos'
            proprio_dim = 14
        # if 'rel' in args.wandb_name or args.model_type == 'discrete':
        hdf5_files = get_hdf5s(args.train_metas_path)
        print('len(hdf5_files)', len(hdf5_files))
        if 'rel' in args.wandb_name:
            data_type = 'rel'
        else:
            data_type = 'abs'
        stats_file = args.train_metas_path.replace(".jsonl", "_global_stats_" + data_type + ".npz")
        print('Saving stats_file', args.train_metas_path, stats_file)
        print('Processing stats for', args.model_type, data_type, control)
        if 'real' in args.wandb_name:
            env = 'real'
        else:
            env = 'sim'
    if 'ACT' in args.model_type:
        model = make_policy('ACT')
    else:
        model, _ = create_model(args.model, proprio_dim=proprio_dim, action_dim=proprio_dim, normalize_proprio=args.normalize_proprio, normalize_action=args.normalize_action)
    if args.pretrained is not None:
        accelerator.print('>>>>>> load pretrain from {}'.format(args.pretrained))
        print(model.load_state_dict(load_file(args.pretrained), strict=False))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    accelerator.print(f'number of params: {n_parameters} M')

    stats = compute_mean_std(hdf5_files, control=control, data_type=data_type, env=env)
    model.normalizer.set_dataset_stats(
        mean={
            "proprio": stats["proprio"]["mean"],
            "action": stats["action"]["mean"],
        },
        std={
            "proprio": stats["proprio"]["std"],
            "action": stats["action"]["std"],
        }
    )
    
    train_dataloader = iter(create_dataloader(
        rank = args.rank,
        world_size = args.world_size,
        batch_size = args.batch_size,
        metas_path = args.train_metas_path,
        num_actions= model.num_action_chunk,
        model_type=args.model_type,
        num_bins=args.num_bins,
        pt_path = args.pt_path,
        normalizer = model.normalizer
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
            # accelerator.save_state(os.path.join(output_dir, f"last_checkpoint"),
            #                        safe_serialization=True)
            accelerator.save_state(os.path.join(output_dir, f"ckpt-{iters}"), 
                                    safe_serialization=True)
            model.train()
        accelerator.wait_for_everyone()
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