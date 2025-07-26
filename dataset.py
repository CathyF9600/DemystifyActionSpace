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

def euler_xyz_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_euler('xyz', q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def convert_hdf5_to_json(hdf5_path): # for RoboTwin
    # 替换基础目录
    json_path = hdf5_path.replace("/downloaded_data/", "/instruction_data/")
    
    # 替换中间路径和扩展名
    json_path = json_path.replace("/data/episode", "/instructions/episode")
    json_path = json_path.replace(".hdf5", ".json")
    
    return json_path

from pathlib import Path

def get_task_name(hdf5_path): # for RoboTwin
    p = Path(hdf5_path)
    # 找到/downloaded_data/的父目录
    for parent in p.parents:
        if parent.name == "downloaded_data":
            return p.relative_to(parent).parts[0]
    return None

class InfiniteDataReader(IterableDataset):
    def __init__(self,
                 rank:int,
                 world_size:int,
                 metas_path:str,
                 num_actions = 10,
                 num_views = 3
                 ):
        #### read meta files, please put all json file in a one directory（metas_path）
        self.rank = rank
        self.world_size = world_size
        self.metas = {}
        self.num_actions = num_actions
        self.num_views = num_views
        # reading setting
        if fileio.isdir(metas_path): meta_files = fileio.list_dir_or_file(metas_path, suffix='.json', recursive=True, list_dir=False)
        else: meta_files, metas_path = [metas_path], ""
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
                meta = json.load(f)
                print(f"================detect dataset {meta['dataset_name']} with traj {len(meta['datalist'])}==================")
                random.shuffle(meta['datalist'])
                self.metas[meta['dataset_name']] = meta
                
        # augmentations
        self.image_aug = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale = (0.8, 1.0),ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        # with open("/mnt/petrelfs/zhengjinliang/HeteroDiffusionPolicy/HeteroFlowPolicy/datasets/utils/new_playtable.json", "r") as f:
        #     self.language_aug = json.load(f)
        self.language_aug = None

    def read_hdf5(self, dataset_name, idx):
        meta = self.metas[dataset_name]
        datapath = meta['datalist'][idx]
        if not isinstance(datapath, str): datapath = datapath[0]
        
        # with io.BytesIO(fileio.get(datapath)) as f:
        #     data = h5py.File(f,'r')
        with h5py.File(datapath, "r") as data:
            ### process actions for robomind (defaultly downsample with a rate of 3)
            images = [data[key] for key in meta['observation_key']] 
            # ins = data[meta["language_instruction_key"]][()].decode() if len(data[meta["language_instruction_key"]].shape) == 0 else \
            #         data[meta["language_instruction_key"]][0].decode()
            
            if dataset_name == 'robotwin2_abs_ee':
                freq = 30  # adjust if needed
                left_ee = data["endpose/left_endpose"][()]      # shape (T, 7)
                right_ee = data["endpose/right_endpose"][()]    # shape (T, 7)
                # 重新排列列：[x,y,z,w,rx,ry,rz] → [x,y,z,rx,ry,rz,w]
                # left_ee = np.column_stack((
                #     left_ee_o[:, :3],  # x,y,z
                #     left_ee_o[:, 4:],  # rx,ry,rz
                #     left_ee_o[:, 3]    # w
                # ))

                # right_ee = np.column_stack((
                #     right_ee_o[:, :3],
                #     right_ee_o[:, 4:],
                #     right_ee_o[:, 3]
                # ))
                left_grip = data["endpose/left_gripper"][()]    # shape (T,)
                right_grip = data["endpose/right_gripper"][()]  # shape (T,)
                left_grip = 1 - left_grip * 2
                right_grip = 1 - right_grip * 2
                action_seq = np.concatenate([
                    left_ee[:, :3],
                    quat_to_rotate6D(left_ee[:, 3:]),                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_ee[:, :3],
                    quat_to_rotate6D(right_ee[:, 3:]),                       # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                # Create zero padding same shape as abs_eef for compatibility
                # abs_eef = np.concatenate([abs_eef], axis=-1)  # now (T, 32)
                # print("abs_eef", abs_eef)
                # import pdb; pdb.set_trace()
                index_list = list(range(0, action_seq.shape[0] - 15))  # or adjust your window length as needed

            elif dataset_name == 'robotwin2_abs_qpos':
                freq = 30  # adjust if needed
                left_joint = data["joint_action/left_arm"][()]      # shape (T, 7)
                right_joint = data["joint_action/right_arm"][()]    # shape (T, 7)
                left_grip = data["joint_action/left_gripper"][()]    # shape (T,)
                right_grip = data["joint_action/right_gripper"][()]  # shape (T,)
                left_grip = 1 - left_grip * 2
                right_grip = 1 - right_grip * 2
                action_seq = np.concatenate([
                    left_joint,                        # (T,7)
                    left_grip[:, None],             # (T,1)
                    right_joint,                    # (T,7)
                    right_grip[:, None]             # (T,1)
                ], axis=-1)
                print('********* action_seq.shape *********', action_seq.shape, left_joint.shape, left_grip.shape)
                index_list = list(range(0, action_seq.shape[0] - 15))  # or adjust your window length as needed
            # elif dataset_name == 'robotwin2_abs_ee':

            else: raise NotImplementedError
            
            random.shuffle(index_list)
            json_p = convert_hdf5_to_json(datapath)
            # print('json_p', json_p)
            with open(json_p, "r") as f:
                self.language_aug = json.load(f)
            for idx in index_list:
                ins = random.choice(self.language_aug["seen"])
                image_input =  torch.stack([self.image_aug(decode_image_from_bytes(img[idx])) for img in images])
                if image_input.size(0) < self.num_views: image_input = torch.cat([image_input, image_input.new_zeros(self.num_views-image_input.size(0), *image_input.shape[1:])], dim=0) 
                action = action_seq[idx:min(idx+freq, action_seq.shape[0])]
                
                action = interp1d(np.arange(len(action)), action, axis=0)(np.linspace(0, len(action)-1, self.num_actions))
                # images: torch.Tensor, # B V C H W
                # encoded_language: torch.Tensor, # B C
                # proprio: torch.Tensor,
                # actions: torch.Tensor 
                items = {
                    # 'hetero_info': torch.tensor(meta['domain_id']),# only 1 domain for this project
                    'images': image_input,
                    'language_instruction': ins,
                    'action_seq': torch.tensor(action).to(torch.float32)
                    }
                
                yield items
               
    def get_generator(self, dataset_name, idx):
         if dataset_name == "AGIBOT": return iter(self.read_lerobot(dataset_name, idx))
         else: return iter(self.read_hdf5(dataset_name, idx))
    
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
                    try: return next(generators[i])
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
