from mmengine import fileio
import numpy as np
import io
import h5py
import json
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
import pyarrow.parquet as pq
from PIL import Image
import av
import random
import math
import cv2
from scipy.spatial.transform import Rotation as R
import os
import re

def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)): camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
    rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
    if rgb is None: 
        rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8) 
        if rgb.size == 2764800: 
            rgb = rgb.reshape(720, 1280, 3) 
        elif rgb.size == 921600: 
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def cal_delta_rotate(q1, q2):
    q1 = R.from_quat(q1)
    q2 = R.from_quat(q2)
    del_rotate = q1 * q2.inv()
    return del_rotate.as_matrix()[..., :, :2].reshape(q1.as_quat().shape[:-1] + (6,))

class InfiniteDataReader(IterableDataset):
    def __init__(self,
                 rank:int,
                 world_size:int,
                 metas_path:str,
                 model_type:str,
                 num_actions = 10,
                 num_bins = 256,
                 pt_path = "encoded_language.pt",
                 ):
        #### read meta files, please put all json file in a one directory（metas_path）
        self.rank = rank
        self.discretize = False
        self.num_bins = num_bins
        self.world_size = world_size
        self.metas = {}
        self.num_actions = num_actions
        if model_type == 'discrete': self.discretize = True
        # reading setting
        with io.BytesIO(fileio.get(metas_path)) as f:
            meta = json.load(f)
            print(f"================detect dataset {meta['dataset_name']} with traj {len(meta['datalist'])}==================")
            random.shuffle(meta['datalist'])
            self.metas[meta['dataset_name']] = meta
        # augmentations
        self.image_aug = transforms.Compose([
            # transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop((224, 224), scale = (0.8, 1.0),ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        self.language_emb = torch.load(pt_path, map_location="cpu")
        print('metas_path', metas_path)
        if 'rel' in metas_path:
            data_type = 'rel'
        else:
            data_type = 'abs'
        stats_file = fileio.join_path(metas_path).replace(".jsonl", "_global_stats_" + data_type + ".npz")


    def quantize_action(self, action):
        # Normalize to [0, 1] using precomputed global min/max
        # print('Normalizing in quantize_action...')
        action = (action - self.global_min[None, :]) / (self.global_max[None, :] - self.global_min[None, :] + 1e-8)
        action = np.clip(action, 0, 1)
        return (action * (self.num_bins - 1)).astype(np.int64)

    def quantize_action_p(self, action):
        """
        Normalize to [0,1] using 5th percentile as min and 95th percentile as max,
        then quantize into bins.
        """
        # print('Normalizing in quantize_action with p5 p95...')
        # normalize with percentile range
        action = (action - self.p5[None, :]) / (self.p95[None, :] - self.p5[None, :] + 1e-8)
        action = np.clip(action, 0, 1)

        return (action * (self.num_bins - 1)).astype(np.int64)

    def read_hdf5(self, dataset_name, idx):
        meta = self.metas[dataset_name]
        datapath = meta['datalist'][idx]
        if not isinstance(datapath, str): datapath = datapath[0]
        rel = False
        with h5py.File(datapath, "r") as data:
            images = [data[key] for key in meta['observation_key']] 
            if dataset_name == 'robotwin2_abs_ee':
                freq = self.num_actions  # adjust if needed
                left_ee = data["endpose/left_endpose"][()]      # shape (T, 7)
                right_ee = data["endpose/right_endpose"][()]    # shape (T, 7)
                left_grip = data["endpose/left_gripper"][()]    # shape (T,)
                right_grip = data["endpose/right_gripper"][()]  # shape (T,)
                prorpio_seq = np.concatenate([
                    left_ee[:, :3],
                    quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_ee[:, :3],
                    quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                action_seq = prorpio_seq[1:]
                # if not self.discretize: # only do min max normalization (later) for discrete model
                #     action_seq = (action_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                index_list = list(range(0, action_seq.shape[0] - self.num_actions))  # or adjust your window length as needed
                # print('action_seq', action_seq[:10])
            elif dataset_name == 'robotwin2_abs_qpos': # 14
                freq = self.num_actions  # adjust if needed
                left_joint = data["joint_action/left_arm"][()]      # shape (T, 7)
                right_joint = data["joint_action/right_arm"][()]    # shape (T, 7)
                left_grip = data["joint_action/left_gripper"][()]    # shape (T,)
                right_grip = data["joint_action/right_gripper"][()]  # shape (T,)
                prorpio_seq = np.concatenate([
                    left_joint,                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_joint,                    # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                action_seq = prorpio_seq[1:]
                # if not self.discretize:
                #     # print('mean std normalizing abs_qpos')
                #     action_seq = (action_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                index_list = list(range(0, action_seq.shape[0] - self.num_actions))  # or adjust your window length as needed
            elif dataset_name == 'robotwin2_rel_ee':
                freq = self.num_actions  # adjust if needed
                left_ee = data["endpose/left_endpose"][()]      # shape (T, 7)
                right_ee = data["endpose/right_endpose"][()]    # shape (T, 7)
                left_grip = data["endpose/left_gripper"][()]    # shape (T,)
                right_grip = data["endpose/right_gripper"][()]  # shape (T,)
                prorpio_seq = np.concatenate([
                    left_ee[:, :3],
                    quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_ee[:, :3],
                    quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                left_delta_xyz = left_ee[1:, :3] - left_ee[:-1, :3]
                right_delta_xyz = right_ee[1:, :3] - right_ee[:-1, :3]
                left_delta_rot6d = cal_delta_rotate(left_ee[1:, 3:], left_ee[:-1, 3:])
                right_delta_rot6d = cal_delta_rotate(right_ee[1:, 3:], right_ee[:-1, 3:])
                ee_diff = np.concatenate([
                    left_delta_xyz,
                    left_delta_rot6d,
                    left_grip[1:, None],   # future gripper value
                    right_delta_xyz,
                    right_delta_rot6d,
                    right_grip[1:, None]
                ], axis=-1)
                # if not self.discretize: # only do min max normalization (later) for discrete model
                #     action_seq = (ee_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # else:
                action_seq = ee_diff
                index_list = list(range(0, action_seq.shape[0] - freq))
            elif dataset_name == 'robotwin2_rel_qpos':
                freq = self.num_actions  # adjust if needed
                left_joint = data["joint_action/left_arm"][()]      # shape (T, 7)
                right_joint = data["joint_action/right_arm"][()]    # shape (T, 7)
                left_grip = data["joint_action/left_gripper"][()]    # shape (T,)
                right_grip = data["joint_action/right_gripper"][()]  # shape (T,)
                prorpio_seq = np.concatenate([
                    left_joint,                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_joint,                    # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                joint_diff = np.concatenate([
                    left_joint[1:] - left_joint[:-1],
                    left_grip[1:, None],  # use future value directly
                    right_joint[1:] - right_joint[:-1],
                    right_grip[1:, None]
                ], axis=-1)
                # if not self.discretize:
                #     action_seq = (joint_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # else:
                action_seq = joint_diff
                index_list = list(range(0, action_seq.shape[0] - freq))  # or adjust your window length as needed
            elif dataset_name == 'real_abs_ee':
                freq = self.num_actions  # adjust if needed
                prorpio_seq = data['observations/eef_6d'][()]
                # print('prorpio_seq', prorpio_seq)
                action_seq = prorpio_seq[1:]
                # if not self.discretize: # only do min max normalization (later) for discrete model
                #     action_seq = (action_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                index_list = list(range(0, action_seq.shape[0] - self.num_actions))  # or adjust your window length as needed
            elif dataset_name == 'real_abs_qpos':
                freq = self.num_actions  # adjust if needed
                prorpio_seq = data['observations/qpos'][()]
                # print('prorpio_seq', prorpio_seq)
                action_seq = prorpio_seq[1:]
                # if not self.discretize: # only do min max normalization (later) for discrete model
                #     action_seq = (action_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                index_list = list(range(0, action_seq.shape[0] - self.num_actions))  # or adjust your window length as needed
            elif dataset_name == 'real_rel_ee':
                freq = self.num_actions  # adjust if needed
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
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                left_delta_xyz = left_ee[1:, :3] - left_ee[:-1, :3]
                right_delta_xyz = right_ee[1:, :3] - right_ee[:-1, :3]
                left_delta_rot6d = cal_delta_rotate(left_ee[1:, 3:], left_ee[:-1, 3:])
                right_delta_rot6d = cal_delta_rotate(right_ee[1:, 3:], right_ee[:-1, 3:])
                ee_diff = np.concatenate([
                    left_delta_xyz,
                    left_delta_rot6d,
                    left_grip[1:, None],   # future gripper value
                    right_delta_xyz,
                    right_delta_rot6d,
                    right_grip[1:, None]
                ], axis=-1)
                # if not self.discretize: # only do min max normalization (later) for discrete model
                #     action_seq = (ee_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # else:
                action_seq = ee_diff
                index_list = list(range(0, action_seq.shape[0] - freq))
            elif dataset_name == 'real_rel_qpos':
                freq = self.num_actions  # adjust if needed
                prorpio_seq = data['observations/qpos'][()]
                left_joint = prorpio_seq[:, :6]
                right_joint = prorpio_seq[:, 7:13]
                left_grip = prorpio_seq[:, 6]
                right_grip = prorpio_seq[:, 13]
                # prorpio_seq = (prorpio_seq - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                joint_diff = np.concatenate([
                    left_joint[1:] - left_joint[:-1],
                    left_grip[1:, None],  # use future value directly
                    right_joint[1:] - right_joint[:-1],
                    right_grip[1:, None]
                ], axis=-1)
                # print('joint_diff')
                # min_val = joint_diff.min(axis=0)
                # max_val = joint_diff.max(axis=0)
                # print('joint_diff min max', min_val, max_val)
                # if not self.discretize:
                #     action_seq = (joint_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                # else:
                action_seq = joint_diff
                index_list = list(range(0, action_seq.shape[0] - freq))  # or adjust your window length as needed

            else: raise NotImplementedError
            
            random.shuffle(index_list)
            for idx in index_list:
                if 'real' in dataset_name:
                    # lang = data["language_instruction"][()] # gpt generated
                    def process_name(path):
                        # get parent folder name
                        folder = os.path.basename(os.path.dirname(path))
                        # drop the trailing underscore + digits (e.g. _0819)
                        folder = re.sub(r'_\d+$', '', folder)
                        # replace underscores with spaces
                        return folder.replace('_', ' ')
                    ins = process_name(datapath)
                    if 'batch2' in ins: ins = ins.replace(' batch2', '')
                    if 'batch3' in ins: ins = ins.replace(' batch3', '')
                    if 'batch4' in ins: ins = ins.replace(' batch4', '')
                else:
                    ins = datapath.split('/')[-3].replace('_', ' ')  # -4, -3, -2 for robotwin sim, real
                # print('ins', ins)
                image_input =  torch.stack([self.image_aug(decode_image_from_bytes(img[idx])) for img in images])
                action = action_seq[idx:idx+self.num_actions]
                # print('self.language_emb', self.language_emb.keys())
                if self.discretize:
                    # print('original action', action[:5])
                    q_action = self.quantize_action_p(action)
                    action_tensor = torch.tensor(q_action, dtype=torch.long)
                    # print('quantized action', q_action[:5])
                else:
                    action_tensor = torch.tensor(action, dtype=torch.float32)
                items = {
                        'images': image_input,
                        'encoded_language': self.language_emb[ins],
                        'action_seq': action_tensor,
                        'proprio': torch.tensor(prorpio_seq[idx]).to(torch.float32)
                        }
                yield items
               
    def get_generator(self, dataset_name, idx):
        return iter(self.read_hdf5(dataset_name, idx))
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.rank = self.rank * worker_info.num_workers + worker_info.id
            self.world_size *= worker_info.num_workers
        dataset_names = list(self.metas.keys())
        idx = [int(self.rank % len(self.metas[dataset_name]['datalist'])) for dataset_name in dataset_names]
        generators = [self.get_generator(dataset_name, i) 
                      for dataset_name, i in zip(dataset_names, idx)]
        while True:
            for i in range(len(generators)):
                def get_next_item():
                    try: return next(generators[i])
                    except StopIteration:
                        idx[i] = (idx[i] + self.world_size) % len(self.metas[dataset_names[i]]['datalist'])
                        generators[i] = self.get_generator(dataset_names[i], int(idx[i]))
                        return get_next_item()
                yield get_next_item()

class ACTWrapperDataset(torch.utils.data.IterableDataset):
    def __init__(self, base_dataset, max_action_len):
        self.base_dataset = base_dataset
        self.max_action_len = max_action_len

    def __iter__(self):
        for items in self.base_dataset:
            image_data = items['images']              # (cams, C, H, W)
            qpos_data = items['proprio']              # (proprio_dim,)
            action_data = items['action_seq']

            yield image_data, qpos_data, action_data, is_pad

def create_dataloader(
                 rank:int,
                 world_size:int,
                 batch_size: int,
                 metas_path:str,
                 num_actions,
                 model_type,
                 num_bins,
                 pt_path
                 ):
    return DataLoader(
            InfiniteDataReader(
                 rank = rank,
                 world_size = world_size,
                 metas_path = metas_path,
                 num_actions = num_actions,
                 model_type = model_type,
                 num_bins = num_bins,
                 pt_path = pt_path
                 ),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
    )


    