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
from dataset import create_dataloader, build_robotwin2_rel_ee_chunk_delta
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
import re
from typing import Optional
from scipy.spatial.transform import Rotation as R

_CNT_MINMAX_MODELS = frozenset(
    {
        "abs_ee_cnt_minmax_rot",
        "abs_qpos_cnt_minmax",
        "rel_ee_cnt_minmax_rot",
        "rel_qpos_cnt_minmax",
    }
)
_CNT_NONE_MODELS = frozenset(
    {
        "abs_ee_cnt_rot",
        "abs_qpos_cnt",
        "rel_ee_cnt_rot",
        "rel_qpos_cnt",
    }
)


def cnt_normalization_mode(model_name: str) -> Optional[str]:
    """连续 cnt 对比：mean / minmax(原始 p5–p95) / none；其它模型返回 None。"""
    if model_name in _CNT_MINMAX_MODELS:
        return "minmax"
    if model_name in _CNT_NONE_MODELS:
        return "none"
    if "_cnt_mean" in model_name:
        return "mean"
    return None


def normalizer_looks_fitted_from_checkpoint(unwrap_model) -> bool:
    """
    resume 加载后：若 ckpt 里已有 dataset 统计量，则不必再跑 compute_mean_std。
    优先看 _stats_fitted；旧 ckpt 无该字段时用 proprio/action mean 是否仍为全 0 推断。
    """
    n = unwrap_model.normalizer
    if hasattr(n, "_stats_fitted") and n._stats_fitted.item() == 1:
        return True
    if n.proprio_mean.abs().sum().item() > 1e-12 or n.action_mean.abs().sum().item() > 1e-12:
        return True
    return False


def _strip_module_prefix_state_dict(sd):
    if not sd:
        return sd
    k0 = next(iter(sd.keys()))
    if k0.startswith("module."):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def _is_missing_key_state_dict_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return "missing key" in err or "error(s) in loading state_dict" in err


def parse_checkpoint_iter_from_path(path: str) -> int:
    """从路径名解析迭代数：ckpt-20000、checkpoint_20000、或最后一级为纯数字目录名。"""
    s = str(path).rstrip("/")
    m = re.search(r"(?:ckpt|checkpoint|iter|step)[_-]?(\d+)", s, re.I)
    if m:
        return int(m.group(1))
    last = Path(s).name
    if last.isdigit() and len(last) >= 3:
        return int(last)
    return 0


def _scalar_to_int(x) -> int:
    if x is None:
        return 0
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def read_global_step_from_accelerator(accelerator) -> int:
    """从 Accelerator 读取已恢复的全局 step（load_state 后）。"""
    s = getattr(accelerator, "step", None)
    if s is not None:
        try:
            v = _scalar_to_int(s)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    st = getattr(accelerator, "state", None)
    if st is not None:
        s2 = getattr(st, "step", None)
        if s2 is not None:
            try:
                v = _scalar_to_int(s2)
                if v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return 0


def read_step_from_resume_rng_pickle(resume_path: str, process_index: int) -> int:
    """
    直接从 checkpoint 目录里的 random_states_*.pkl 读 step（与 Accelerate save_state 一致）。
    load_state 后若 accelerator.step 未写入，此方式最可靠。
    """
    base = Path(resume_path).expanduser().resolve()
    candidates = []
    f = base / f"random_states_{process_index}.pkl"
    if f.exists():
        candidates.append(f)
    try:
        for p in sorted(base.glob("random_states_*.pkl")):
            if p not in candidates:
                candidates.append(p)
    except OSError:
        pass
    for p in candidates:
        try:
            states = torch.load(p, map_location="cpu", weights_only=False)
            if not isinstance(states, dict):
                continue
            if "step" not in states:
                continue
            v = _scalar_to_int(states["step"])
            if v > 0:
                return v
        except Exception:
            continue
    return 0


def infer_start_iter_from_sibling_ckpts(resume_path: str, save_interval: int) -> int:
    """
    resume 指向 ckpt-final（路径名含 final）且本目录名里抽不出数字时：
    取**同一父目录下**其它 ckpt-<数字> 子目录的最大数字 N，推断 ckpt-final 对应训练已走到
    **N + save_interval**（最后一次按间隔落盘之后，又训到结束再存 final；与 train 里
    「先 ckpt-N 再训若干步再 ckpt-final」一致）。
    若无兄弟 ckpt-* 或路径名不含 final，返回 0。
    """
    rp = Path(resume_path).expanduser().resolve()
    if "final" not in rp.name.lower():
        return 0
    parent = rp.parent
    best = 0
    try:
        for p in parent.iterdir():
            if not p.is_dir():
                continue
            m = re.match(r"^ckpt-(\d+)$", p.name)
            if m:
                best = max(best, int(m.group(1)))
    except OSError:
        pass
    if best <= 0:
        return 0
    si = max(0, int(save_interval))
    return best + si


def get_training_start_iter(
    accelerator, resume_path: Optional[str], save_interval: int
) -> int:
    """
    resume 后应从 global step 继续，而不是从 0。
    顺序：Accelerator.step -> random_states_*.pkl -> 本路径名中的数字 ->
    ckpt-final 时同目录 ckpt-<N> 推断为 N+save_interval。
    """
    step = read_global_step_from_accelerator(accelerator)
    if step > 0:
        return step
    if resume_path:
        step = read_step_from_resume_rng_pickle(resume_path, accelerator.process_index)
        if step > 0:
            return step
        parsed = parse_checkpoint_iter_from_path(resume_path)
        if parsed > 0:
            return parsed
        sib = infer_start_iter_from_sibling_ckpts(resume_path, save_interval)
        if sib > 0:
            return sib
    return 0


def load_resume_relaxed(accelerator, model, optim, resume_path: str):
    """
    优先 accelerator.load_state(..., strict=False)，兼容旧 checkpoint 无 normalizer._stats_fitted 等 key。
    若 accelerate 版本不支持 strict 或仍失败，则对模型权重使用 strict=False 手动加载，并尽量恢复 optimizer/rng。
    """
    try:
        accelerator.load_state(resume_path, strict=False)
        return
    except TypeError:
        # 旧版 accelerate 无 strict 参数
        try:
            accelerator.load_state(resume_path)
            return
        except RuntimeError as e:
            if not _is_missing_key_state_dict_error(e):
                raise
            accelerator.print(
                ">>>>>>>> resume: 旧 checkpoint 与当前模型 key 不完全一致，改为 strict=False 手动加载"
            )
            _manual_load_resume_weights(accelerator, model, optim, resume_path)
    except RuntimeError as e:
        if not _is_missing_key_state_dict_error(e):
            raise
        accelerator.print(
            ">>>>>>>> resume: 旧 checkpoint 与当前模型 key 不完全一致，改为 strict=False 手动加载"
        )
        _manual_load_resume_weights(accelerator, model, optim, resume_path)


def _accelerator_is_main(accelerator) -> bool:
    if hasattr(accelerator, "is_main_process"):
        return bool(accelerator.is_main_process)
    return int(os.environ.get("RANK", 0)) == 0


def _renamed_final_flag_after_barrier(resume_path: str, start_iter: int) -> bool:
    """各 rank 在 barrier 后根据路径是否由 ckpt-final 变为 ckpt-{start_iter} 得到统一布尔值。"""
    if start_iter <= 0:
        return False
    rp = Path(resume_path).expanduser().resolve()
    if "final" not in rp.name.lower():
        return False
    dst = rp.parent / f"ckpt-{start_iter}"
    return dst.is_dir() and (not rp.exists())


def rename_ckpt_final_after_resume(
    accelerator, resume_path: str, start_iter: int
) -> bool:
    """
    将上一轮保存的 ckpt-final 重命名为 ckpt-{start_iter}（与「final = 训练终点步」一致），
    这样本轮下一个周期性目录将是 ckpt-{start_iter+save_interval}（例如 20k→30k，save_interval=10k）。
    仅主进程执行 rename，随后 wait_for_everyone，各 rank 用 _renamed_final_flag_after_barrier 对齐结果。
    """
    if start_iter <= 0:
        accelerator.wait_for_everyone()
        return False
    rp = Path(resume_path).expanduser().resolve()
    if "final" not in rp.name.lower() or not rp.is_dir():
        accelerator.wait_for_everyone()
        return False
    dst = rp.parent / f"ckpt-{start_iter}"
    if _accelerator_is_main(accelerator):
        if dst.exists():
            accelerator.print(
                f">>>>>>>> skip rename：已存在 {dst.name}，保留原 ckpt-final 或手动处理"
            )
        else:
            try:
                old_name = rp.name
                rp.rename(dst)
                accelerator.print(
                    f">>>>>>>> 已 rename {old_name} -> {dst.name}（释放 ckpt-final 名供本轮结束写入）"
                )
            except OSError as e:
                accelerator.print(f">>>>>>>> rename 失败: {e}")
    accelerator.wait_for_everyone()
    return _renamed_final_flag_after_barrier(resume_path, start_iter)


def _manual_load_resume_weights(accelerator, model, optim, resume_path: str):
    unwrapped = accelerator.unwrap_model(model)
    base = Path(resume_path)
    if (base / "model.safetensors").exists():
        sd = load_file(str(base / "model.safetensors"))
    elif (base / "pytorch_model.bin").exists():
        sd = torch.load(base / "pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"resume 目录中未找到 model.safetensors 或 pytorch_model.bin: {resume_path}"
        )
    sd = _strip_module_prefix_state_dict(sd)
    unwrapped.load_state_dict(sd, strict=False)
    opt_f = base / "optimizer.bin"
    if opt_f.exists():
        try:
            optim.load_state_dict(
                torch.load(opt_f, map_location="cpu", weights_only=False)
            )
        except Exception as ex:
            accelerator.print(f">>>>>>>> 警告: optimizer 状态未完全恢复 ({ex})")
    rng_f = base / f"random_states_{accelerator.process_index}.pkl"
    if rng_f.exists():
        try:
            states = torch.load(rng_f, map_location="cpu", weights_only=False)
            if "random_state" in states:
                random.setstate(states["random_state"])
            if "numpy_random_seed" in states:
                np.random.set_state(states["numpy_random_seed"])
            if "torch_manual_seed" in states:
                torch.set_rng_state(states["torch_manual_seed"])
            if "step" in states:
                try:
                    accelerator.step = states["step"]
                except (AttributeError, TypeError):
                    try:
                        setattr(accelerator, "step", states["step"])
                    except Exception:
                        pass
        except Exception as ex:
            accelerator.print(f">>>>>>>> 警告: RNG/step 状态未完全恢复 ({ex})")


def apply_dataset_stats_to_model(model, stats, norm_mode: Optional[str]):
    """将 compute_mean_std 的 stats 写入 model.normalizer。"""
    if norm_mode == "minmax":
        model.normalizer.set_dataset_stats(
            mean={
                "proprio": stats["proprio"]["mean"],
                "action": stats["action"]["mean"],
            },
            std={
                "proprio": stats["proprio"]["std"],
                "action": stats["action"]["std"],
            },
            proprio_p5=stats["proprio"]["p5"],
            proprio_p95=stats["proprio"]["p95"],
            action_p5=stats["action"]["p5"],
            action_p95=stats["action"]["p95"],
        )
    else:
        model.normalizer.set_dataset_stats(
            mean={
                "proprio": stats["proprio"]["mean"],
                "action": stats["action"]["mean"],
            },
            std={
                "proprio": stats["proprio"]["std"],
                "action": stats["action"]["std"],
            },
        )


def get_args_parser():
    parser = argparse.ArgumentParser('Training script', add_help=False)
    # Base Settings
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument(
        '--iters',
        default=1000000,
        type=int,
        help='训练 global 的结束步数（不含）：循环为 range(start_iter, iters)。'
        ' resume 时 start_iter 来自 checkpoint（或路径 ckpt-XXXXX），例如 --iters 40000 表示从 20k 续训到 40k。',
    )
    parser.add_argument('--train_metas_path', type=str)
    parser.add_argument('--pt_path', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--precision', default='no', type=str)
    
    parser.add_argument('--model', default='model_base', type=str)
    parser.add_argument('--delta_type', type=str, default="chunk", choices=["chunk", "step"], help="Delta type")
    parser.add_argument('--rot_repr', type=str)

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
    
    parser.add_argument(
        '--resume',
        default=None,
        help='Accelerate 完整 checkpoint 目录（含 optimizer 等）；若指定则忽略 --pretrained',
    )
    parser.add_argument(
        '--pretrained',
        default=None,
        help='仅加载权重（safetensors），用于无 --resume 时的初始化',
    )
    parser.add_argument(
        '--force-recompute-stats',
        action='store_true',
        help='即使 resume 也重新扫 HDF5 计算 mean/std/p 分位数并覆盖 normalizer',
    )
    parser.add_argument(
        '--start_iter',
        default=-1,
        type=int,
        help='仅在 --resume 时生效：全局起始 iter，覆盖自动推断。默认 -1 为自动；'
        '例如已从 0 训到 20000 后改训到 40000，可设 --start_iter 20000 --iters 40000。',
    )
    parser.add_argument(
        '--final_ckpt_name',
        default='ckpt-final',
        type=str,
        help='训练结束时 accelerator.save_state 的子目录名；续训时可改为 ckpt-final-40k 以免覆盖上次 ckpt-final。',
    )

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29531, type=int, help='port')

    parser.add_argument('--wandb_name', default='robotwin2_abs_qpos', type=str)
    parser.add_argument('--normalize_action', default=False, action='store_true', help='load ckpt path')
    parser.add_argument('--normalize_proprio', default=False, action='store_true', help='load ckpt path')


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

def convert_rot(quat, rot_repr):
    if rot_repr == "rot6d":
        return quat_to_rotate6D(quat)
    elif rot_repr == "quat":
        return quat
    elif rot_repr == "euler":
        return R.from_quat(quat).as_euler('xyz', degrees=False)
    else:
        raise ValueError

def align_quat(q):
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i-1]) < 0:
            q[i] = -q[i]
    return q

def angle_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_euler('xyz', degrees=False)

def compute_mean_std(hdf5_paths, 
        control='ee', 
        data_type='rel',     
        rot_repr="rot6d",
        chunk_wise=True,
        env=None,
        num_action_chunk=30,
    ):
    all_proprios = []
    all_actions = []
    print('control, data_type, rot_repr, chunk_wise, env', control, data_type, rot_repr, chunk_wise, env)
    for path in hdf5_paths:
        with h5py.File(path, 'r') as data:
            if data_type =='rel':
                if control == 'qpos':
                    if env == 'real':
                        prorpio_seq = data['observations/qpos'][()]
                        left_joint = prorpio_seq[:, :6]
                        right_joint = prorpio_seq[:, 7:13]
                        left_grip = prorpio_seq[:, 6]
                        right_grip = prorpio_seq[:, 13]
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
                        prorpio_seq = np.concatenate([
                            left_joint,
                            left_grip[:, None],
                            right_joint,
                            right_grip[:, None]
                        ], axis=-1)
                        if not chunk_wise:
                            joint_diff = np.concatenate([
                                left_joint[1:] - left_joint[:-1],
                                left_grip[1:, None],
                                right_joint[1:] - right_joint[:-1],
                                right_grip[1:, None]
                            ], axis=-1)
                            action_seq = joint_diff
                        else: # chunk_wise：与 dataset 一致，delta 相对窗口起点 q_idx
                            freq = num_action_chunk
                            T = left_joint.shape[0]
                            rows = []
                            for win_idx in range(0, T - freq):
                                ch = np.concatenate([
                                    left_joint[win_idx + 1 : win_idx + 1 + freq]
                                    - left_joint[win_idx : win_idx + 1],
                                    left_grip[win_idx + 1 : win_idx + 1 + freq, None],
                                    right_joint[win_idx + 1 : win_idx + 1 + freq]
                                    - right_joint[win_idx : win_idx + 1],
                                    right_grip[win_idx + 1 : win_idx + 1 + freq, None],
                                ], axis=-1)
                                rows.append(ch)
                            action_seq = (
                                np.concatenate(rows, axis=0)
                                if rows
                                else np.zeros((0, prorpio_seq.shape[1]))
                            )
                else: # ee
                    if env == 'real':
                        prorpio_seq = data['observations/eef_quaternion'][()]
                        left_ee = prorpio_seq[:, :7]
                        right_ee = prorpio_seq[:, 8:15]
                        left_grip = prorpio_seq[:, 7]
                        right_grip = prorpio_seq[:, 15]
                        prorpio_seq = np.concatenate([
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
                        left_ee = data["endpose/left_endpose"][()]
                        right_ee = data["endpose/right_endpose"][()]
                        left_grip = data["endpose/left_gripper"][()]
                        right_grip = data["endpose/right_gripper"][()]
                        prorpio_seq = np.concatenate([
                            left_ee[:, :3],
                            convert_rot(left_ee[:, 3:], rot_repr),
                            left_grip[:, None],
                            right_ee[:, :3],
                            convert_rot(right_ee[:, 3:], rot_repr),
                            right_grip[:, None]
                        ], axis=-1)
                        if not chunk_wise:
                            left_delta_xyz = left_ee[1:, :3] - left_ee[:-1, :3]
                            right_delta_xyz = right_ee[1:, :3] - right_ee[:-1, :3]
                            # 统一：rotation 全部直接减
                            if rot_repr == "rot6d":
                                left_rot = quat_to_rotate6D(left_ee[:, 3:])
                                right_rot = quat_to_rotate6D(right_ee[:, 3:])
                            elif rot_repr == "quat":
                                left_rot = align_quat(left_ee[:, 3:])
                                right_rot = align_quat(right_ee[:, 3:])
                            elif rot_repr == "euler":
                                left_rot = quat_to_euler(left_ee[:, 3:])
                                right_rot = quat_to_euler(right_ee[:, 3:])
                            # ⚠️ Euler delta 会有 wrap 问题（π → -π）
                            if rot_repr == "euler":
                                left_delta_rot = angle_diff(left_rot[1:], left_rot[:-1])
                                right_delta_rot = angle_diff(right_rot[1:], right_rot[:-1])
                            else:
                                left_delta_rot = left_rot[1:] - left_rot[:-1]
                                right_delta_rot = right_rot[1:] - right_rot[:-1]
                            ee_diff = np.concatenate([
                                left_delta_xyz,
                                left_delta_rot,
                                left_grip[1:, None],
                                right_delta_xyz,
                                right_delta_rot,
                                right_grip[1:, None]
                            ], axis=-1)
                            action_seq = ee_diff
                        else: # chunk-wise：与 dataset 一致，delta 相对窗口起点
                            freq = num_action_chunk
                            T = left_ee.shape[0]
                            rows = []
                            for win_idx in range(0, T - freq):
                                ch = build_robotwin2_rel_ee_chunk_delta(
                                    left_ee,
                                    right_ee,
                                    left_grip,
                                    right_grip,
                                    win_idx,
                                    freq,
                                    rot_repr,
                                )
                                rows.append(ch)
                            action_seq = (
                                np.concatenate(rows, axis=0)
                                if rows
                                else np.zeros((0, prorpio_seq.shape[1]))
                            )

            if data_type == 'abs':
                if control == 'qpos':
                    if env == 'real':
                        action_seq = data['observations/qpos'][()] # 实机
                        prorpio_seq = action_seq
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
                        prorpio_seq = action_seq
                else: # ee
                    if env == 'real':
                        action_seq = data['observations/eef_6d'][()] # 实机
                        prorpio_seq = action_seq
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
                        prorpio_seq = action_seq
            if action_seq.shape[0] > 0:
                all_actions.append(action_seq)
            all_proprios.append(prorpio_seq)
    # ---- Compute stats ----
    if len(all_actions) == 0:
        raise RuntimeError(
            "compute_mean_std: no action rows (check trajectories length vs num_action_chunk)."
        )
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
            "p5": np.percentile(stacked_proprios, 5, axis=0),
            "p95": np.percentile(stacked_proprios, 95, axis=0),
        },
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


def main(args):

    output_dir = Path(args.output_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision = args.precision,
                              log_with="tensorboard", 
                              project_dir=output_dir, kwargs_handlers=[kwargs])
    accelerator.init_trackers("HFP_Training")
    # torch.distributed.barrier()

    cfg_tag = f"{args.wandb_name} {args.model}".lower()
    control = "ee" if "ee" in cfg_tag else "qpos"
    proprio_dim = 20 if control == "ee" else 14

    hdf5_files = get_hdf5s(args.train_metas_path)
    print("len(hdf5_files)", len(hdf5_files))
    data_type = "rel" if "rel" in cfg_tag else "abs"

    config = [args.delta_type, args.rot_repr]

    rot_repr = args.rot_repr

    chunk_wise = any(str(c).startswith("chunk") for c in config)

    model, _ = create_model(
        args.model,
        dim_proprio=proprio_dim,
        dim_actions=proprio_dim,
    )
    if args.resume is not None and args.pretrained is not None:
        accelerator.print(">>>>>> 已指定 --resume，将忽略 --pretrained")
    if args.pretrained is not None and args.resume is None:
        accelerator.print(">>>>>> load pretrain from {}".format(args.pretrained))
        print(model.load_state_dict(load_file(args.pretrained), strict=False))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000 / 1000
    accelerator.print(f"number of params: {n_parameters} M")

    train_dataloader = iter(create_dataloader(
        rank = int(os.environ.get("RANK", 0)),
        # rank = args.rank,
        world_size = args.world_size,
        batch_size = args.batch_size,
        metas_path = args.train_metas_path,
        num_actions= model.num_action_chunk,
        model_type=args.model_type,
        num_bins=args.num_bins,
        pt_path = args.pt_path,
        config=config
    ))
    
    model = model.to(torch.float32)
    # 设置优化器参数组
    optim = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )
    
    model, optim = accelerator.prepare(model, optim)
    start_iter = 0
    renamed_final = False
    if args.resume is not None:
        accelerator.print(">>>>>> resume from {}".format(args.resume))
        load_resume_relaxed(accelerator, model, optim, args.resume)
        start_iter = get_training_start_iter(
            accelerator, args.resume, args.save_interval
        )
        auto_iter = start_iter
        if args.start_iter >= 0:
            start_iter = args.start_iter
            accelerator.print(
                f">>>>>>>> 使用手动 --start_iter={start_iter}（自动推断为 {auto_iter}）"
            )
        accelerator.print(f">>>>>>>> 从 global iter {start_iter} 继续训练（目标 iters={args.iters}）")
        if start_iter == 0 and args.start_iter < 0:
            accelerator.print(
                ">>>>>>>> 警告: 未能解析起始 iter（已尝试 Accelerator.step、random_states_*.pkl、"
                "路径名、ckpt-final 时同目录 ckpt-<N>→N+save_interval）。将从 iter 0 开始；可设 --start_iter。"
            )
        if start_iter > 0:
            renamed_final = rename_ckpt_final_after_resume(
                accelerator, args.resume, start_iter
            )

    unwrapped = accelerator.unwrap_model(model)
    norm_mode = cnt_normalization_mode(args.model)
    need_compute_stats = (
        args.force_recompute_stats
        or args.resume is None
        or not normalizer_looks_fitted_from_checkpoint(unwrapped)
    )
    if need_compute_stats:
        accelerator.print(
            ">>>>>>>> compute_mean_std"
            + (" [force-recompute-stats]" if args.force_recompute_stats else "")
        )
        stats = compute_mean_std(
            hdf5_files,
            control=control,
            data_type=data_type,
            rot_repr=rot_repr,
            chunk_wise=chunk_wise,
            env=args.env,
            num_action_chunk=unwrapped.num_action_chunk,
        )
        apply_dataset_stats_to_model(unwrapped, stats, norm_mode)
    else:
        accelerator.print(
            ">>>>>>>> skip compute_mean_std（normalizer 统计量已从 checkpoint 恢复）"
        )

    train_dataloader = iter(train_dataloader)
    model.train()
    if start_iter >= args.iters:
        accelerator.print(
            f"start_iter={start_iter} >= args.iters={args.iters}，无需训练；退出。"
        )
        return

    remaining = args.iters - start_iter
    accelerator.print(
        f"Start training: global iter {start_iter} .. {args.iters - 1}（共 {remaining} 步）"
    )

    for iters in range(start_iter, args.iters):
        past_time = time.time()
        data = next(train_dataloader)
        inputs = {
            **{key: value.to(accelerator.device, non_blocking=True) for key, value in data.items()},
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
            if renamed_final and iters == start_iter:
                accelerator.print(
                    f">>>>>>>> skip save ckpt-{iters}（已由 ckpt-final rename 得到同内容里程碑）"
                )
            else:
                model.eval()
                accelerator.print("========start saving models=========")
                accelerator.save_state(
                    os.path.join(output_dir, f"ckpt-{iters}"),
                    safe_serialization=True,
                )
                model.train()
        accelerator.wait_for_everyone()
    accelerator.save_state(
        os.path.join(output_dir, args.final_ckpt_name),
        safe_serialization=True,
    )

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
    main(args)