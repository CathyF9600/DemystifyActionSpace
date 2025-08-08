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
                 num_actions = 10,
                 ):
        #### read meta files, please put all json file in a one directory（metas_path）
        self.rank = rank
        self.world_size = world_size
        self.metas = {}
        self.num_actions = num_actions
        # reading setting
        with io.BytesIO(fileio.get(metas_path)) as f:
            meta = json.load(f)
            print(f"================detect dataset {meta['dataset_name']} with traj {len(meta['datalist'])}==================")
            random.shuffle(meta['datalist'])
            self.metas[meta['dataset_name']] = meta
        # augmentations
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        self.language_emb = torch.load("encoded_language.pt", map_location="cpu")

    def read_hdf5(self, dataset_name, idx):
        meta = self.metas[dataset_name]
        datapath = meta['datalist'][idx]
        if not isinstance(datapath, str): datapath = datapath[0]
 
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
                index_list = list(range(0, action_seq.shape[0] - self.num_actions))  # or adjust your window length as needed

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
                action_seq = (ee_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                index_list = list(range(0, action_seq.shape[0] - freq))
            elif dataset_name == 'robotwin2_rel_qpos':
                freq = self.num_actions  # adjust if needed
                left_joint = data["joint_action/left_arm"][()]      # shape (T, 7)
                right_joint = data["joint_action/right_arm"][()]    # shape (T, 7)
                left_grip = data["joint_action/left_gripper"][()]    # shape (T,)
                right_grip = data["joint_action/right_gripper"][()]  # shape (T,)
                left_grip = 1 - left_grip * 2
                right_grip = 1 - right_grip * 2
                prorpio_seq = np.concatenate([
                    left_joint,                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_joint,                    # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                joint_diff = np.concatenate([
                    left_joint[1:] - left_joint[:-1],
                    left_grip[1:, None],  # use future value directly
                    right_joint[1:] - right_joint[:-1],
                    right_grip[1:, None]
                ], axis=-1)
                action_seq = (joint_diff - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
                index_list = list(range(0, action_seq.shape[0] - freq))  # or adjust your window length as needed
            else: raise NotImplementedError
            
            random.shuffle(index_list)
            for idx in index_list:
                ins = datapath.split('/')[-4].replace('_', ' ') 
                image_input =  torch.stack([self.image_aug(decode_image_from_bytes(img[idx])) for img in images])
                action = action_seq[idx:idx+self.num_actions]
                items = {
                        'images': image_input,
                        'encoded_language': self.language_emb[ins],
                        'action_seq': torch.tensor(action).to(torch.float32),
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


def create_dataloader(
                 rank:int,
                 world_size:int,
                 batch_size: int,
                 metas_path:str,
                 num_actions,
                 ):
    return DataLoader(
            InfiniteDataReader(
                 rank = rank,
                 world_size = world_size,
                 metas_path = metas_path,
                 num_actions = num_actions,
                 ),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
    )