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
from scipy.interpolate import interp1d

def read_video_to_frames(data_path):
    buf = io.BytesIO(fileio.get(data_path))
    container = av.open(buf)
    frames = []
    for packet in container.demux(video=0):
        for frame in packet.decode():
            img = frame.to_ndarray(format='rgb24')  
            frames.append(img)
    return np.stack(frames, axis=0)

def decode_image_from_bytes(camera_rgb_image):
    if isinstance(camera_rgb_image, (bytes, bytearray)): 
        camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
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

def euler_xyz_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_euler('xyz', q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


class InfiniteDataReader(IterableDataset):
    def __init__(self,
                 rank:int,
                 world_size:int,
                 metas_path:str,
                 num_actions = 10,
                 num_views = 3
                 ):
        self.rank = rank
        self.world_size = world_size
        self.metas = {}
        self.num_actions = num_actions
        self.num_views = num_views

        if fileio.isdir(metas_path): 
            meta_files = fileio.list_dir_or_file(metas_path, suffix='.json', recursive=True, list_dir=False)
        else: 
            meta_files, metas_path = [metas_path], ""
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
                meta = json.load(f)
                print(f"Detected dataset {meta['dataset_name']} with {len(meta['datalist'])} trajectories")
                random.shuffle(meta['datalist'])
                self.metas[meta['dataset_name']] = meta
                
        self.image_aug = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

    def read_hdf5_abs_eef(self, dataset_name, idx):
        meta = self.metas[dataset_name]
        datapath = meta['datalist'][idx]
        if not isinstance(datapath, str): 
            datapath = datapath[0]
        
        with h5py.File(datapath, "r") as data:
            images = [data[key] for key in meta['observation_key']]
            ins = data[meta["language_instruction_key"]][()].decode() if len(data[meta["language_instruction_key"]].shape) == 0 else \
                  data[meta["language_instruction_key"]][0].decode()

            # Extract proprioceptive data and actions for each dataset
            if dataset_name == 'Calvin':  # Delta EEF single arm
                freq = 30
                proprio_data = data["proprio"][()]  # (T, ...) raw proprio
                abs_eef = np.concatenate([
                    proprio_data[:, :3], 
                    euler_xyz_to_rotate6D(proprio_data[:, 3:6]), 
                    proprio_data[:, -1:]
                ], axis=-1)
                abs_eef = np.concatenate([abs_eef, np.zeros_like(abs_eef)], axis=-1)
                index_list = list(range(0, abs_eef.shape[0] - 15))

            elif dataset_name == 'RT1':
                freq = 30
                proprio_data = np.concatenate([  # Example: combine eef and gripper as proprio
                    data["eef_quat_orientation"][()],
                    data['gripper'][:, None]
                ], axis=-1)
                abs_eef = np.concatenate([
                    proprio_data[:, :3], 
                    quat_to_rotate6D(proprio_data[:, 3:7]),  # quaternion part
                    proprio_data[:, -1:]
                ], axis=-1)
                abs_eef = np.concatenate([abs_eef, np.zeros_like(abs_eef)], axis=-1)
                index_list = list(range(1, abs_eef.shape[0] - 15))

            elif dataset_name == 'libero':
                freq = 30
                proprio_data = data['proprio'][()]  # Adjust based on actual key
                abs_eef = data['abs_action_6d'][:-1]
                abs_eef = np.concatenate([abs_eef, np.zeros_like(abs_eef)], axis=-1)
                index_list = list(range(0, abs_eef.shape[0] - 15))
                images = [img[1:] for img in images]  # Adjust image indices

            elif dataset_name == 'VLABench':
                freq = 30
                proprio_data = data["proprio"][()]
                abs_eef = np.concatenate([
                    proprio_data[:, :3], 
                    euler_xyz_to_rotate6D(proprio_data[:, 3:6]), 
                    1 - proprio_data[:, -1:] * 2
                ], axis=-1)
                abs_eef = np.concatenate([abs_eef, np.zeros_like(abs_eef)], axis=-1)
                index_list = list(range(0, abs_eef.shape[0] - 15))

            elif dataset_name == 'RoboTwin':
                freq = 30
                left_ee = data["endpose/left_endpose_xyzw"][()]
                right_ee = data["endpose/right_endpose_xyzw"][()]
                left_grip = data["endpose/left_gripper"][()]
                right_grip = data["endpose/right_gripper"][()]
                proprio_data = np.concatenate([  # Combine all as proprio
                    left_ee, right_ee, left_grip[:, None], right_grip[:, None]
                ], axis=-1)
                left_grip_norm = 1 - left_grip * 2
                right_grip_norm = 1 - right_grip * 2
                abs_eef = np.concatenate([
                    left_ee[:, :3], quat_to_rotate6D(left_ee[:, 3:]), left_grip_norm[:, None],
                    right_ee[:, :3], quat_to_rotate6D(right_ee[:, 3:]), right_grip_norm[:, None]
                ], axis=-1)
                index_list = list(range(0, abs_eef.shape[0] - 15))

            else: 
                raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
            random.shuffle(index_list)
            
            for idx_t in index_list:
                # Process images (V views)
                image_input = torch.stack([
                    self.image_aug(decode_image_from_bytes(img[idx_t])) 
                    for img in images
                ])
                if image_input.size(0) < self.num_views:
                    image_input = torch.cat([
                        image_input, 
                        image_input.new_zeros(self.num_views - image_input.size(0), *image_input.shape[1:])
                    ], dim=0)

                # Interpolate actions to num_actions steps
                action_slice = abs_eef[idx_t:min(idx_t + freq, abs_eef.shape[0])]
                action = interp1d(
                    np.arange(len(action_slice)), 
                    action_slice, 
                    axis=0
                )(np.linspace(0, len(action_slice)-1, self.num_actions))

                # Get proprio at current timestep
                proprio = proprio_data[idx_t]

                yield {
                    'hetero_info': torch.tensor(meta['domain_id']),
                    'language_instruction': ins,
                    'image_input': image_input,  # (V, C, H, W)
                    'proprio': torch.tensor(proprio).to(torch.float32),  # Added: proprioceptive input
                    'actions': torch.tensor(action).to(torch.float32)  # Renamed from abs_eef
                }

    def read_hdf5_rel_eef(self, dataset_name, idx):
        meta = self.metas[dataset_name]
        datapath = meta['datalist'][idx]
        if not isinstance(datapath, str): 
            datapath = datapath[0]
        
        with h5py.File(datapath, "r") as data:
            images = [data[key] for key in meta['observation_key']]
            ins = data[meta["language_instruction_key"]][()].decode() if len(data[meta["language_instruction_key"]].shape) == 0 else \
                  data[meta["language_instruction_key"]][0].decode()
            if dataset_name == 'RoboTwin':
                freq = 30
                left_ee = data["endpose/left_endpose_xyzw"][()]
                right_ee = data["endpose/right_endpose_xyzw"][()]
                left_grip = data["endpose/left_gripper"][()]
                right_grip = data["endpose/right_gripper"][()]
                proprio_data = np.concatenate([  # Combine all as proprio
                    left_ee, right_ee, left_grip[:, None], right_grip[:, None]
                ], axis=-1)
                left_grip_norm = 1 - left_grip * 2
                right_grip_norm = 1 - right_grip * 2
                abs_eef = np.concatenate([
                    left_ee[:, :3], quat_to_rotate6D(left_ee[:, 3:]), left_grip_norm[:, None],
                    right_ee[:, :3], quat_to_rotate6D(right_ee[:, 3:]), right_grip_norm[:, None]
                ], axis=-1)
                index_list = list(range(0, abs_eef.shape[0] - 15))
            # if dataset_name == :
            
            else: 
                raise NotImplementedError(f"Dataset {dataset_name} not supported")
            
            random.shuffle(index_list)
            
            for idx_t in index_list:
                # Process images (V views)
                image_input = torch.stack([
                    self.image_aug(decode_image_from_bytes(img[idx_t])) 
                    for img in images
                ])
                if image_input.size(0) < self.num_views:
                    image_input = torch.cat([
                        image_input, 
                        image_input.new_zeros(self.num_views - image_input.size(0), *image_input.shape[1:])
                    ], dim=0)

                # Interpolate actions to num_actions steps
                action_slice = abs_eef[idx_t:min(idx_t + freq, abs_eef.shape[0])]
                action = interp1d(
                    np.arange(len(action_slice)), 
                    action_slice, 
                    axis=0
                )(np.linspace(0, len(action_slice)-1, self.num_actions))

                # Get proprio at current timestep
                proprio = proprio_data[idx_t]

                yield {
                    'hetero_info': torch.tensor(meta['domain_id']),
                    'language_instruction': ins,
                    'image_input': image_input,  # (V, C, H, W)
                    'proprio': torch.tensor(proprio).to(torch.float32),  # Added: proprioceptive input
                    'actions': torch.tensor(action).to(torch.float32)  # Renamed from abs_eef
                }

    def get_generator(self, dataset_name, idx):
        if dataset_name == "AGIBOT": 
            return iter(self.read_lerobot(dataset_name, idx))
        else: # absolute control for EE
            return iter(self.read_hdf5_abs_eef(dataset_name, idx))
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.rank = self.rank * worker_info.num_workers + worker_info.id
            self.world_size *= worker_info.num_workers
        dataset_names = list(self.metas.keys())
        idx = [int(self.rank % len(self.metas[dataset_name]['datalist'])) for dataset_name in dataset_names]
        generators = [self.get_generator(dataset_name, i) 
                      for dataset_name, i in zip(dataset_names, idx) ]
        
        while True:
            for i in range(len(generators)):
                def get_next_item():
                    try: 
                        return next(generators[i])
                    except StopIteration:
                        idx[i] = (idx[i] + self.world_size) % len(self.metas[dataset_names[i]]['datalist'])
                        generators[i] = self.get_generator(dataset_names[i], int(idx[i]))
                        return get_next_item()
                    except Exception as e:
                        meta = self.metas[dataset_names[i]]
                        with open("error_data.log", "a+") as f:
                            f.write(f"{meta['datalist'][idx[i]]} :{e}\n")
                        print(meta['datalist'][idx[i]], f':{e}')
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
                 num_views = 3
                 ):
    return DataLoader(
            InfiniteDataReader(
                 rank = rank,
                 world_size = world_size,
                 metas_path = metas_path,
                 num_actions = num_actions,
                 num_views = num_views
                 ),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
    )