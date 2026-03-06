from mmengine import fileio
import numpy as np
import io
import h5py
import json
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from PIL import Image
import random
import math
import cv2
from scipy.interpolate import interp1d


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


class MapstyleDataReader(Dataset):
    def __init__(self,
                 metas_path:str,
                 num_actions = 10, 
                 sample_num = 100000
                 ):
        #### read meta files, please put all json file in a one directory（metas_path）
        self.num_actions = num_actions
        # reading setting
        self.datalist = []
        self.database = {}
        readed_tasks_num = {}
        if fileio.isdir(metas_path): meta_files = fileio.list_dir_or_file(metas_path, suffix='.json', recursive=True, list_dir=False)
        else: meta_files, metas_path = [metas_path], ""
        
        for file in meta_files:
            with io.BytesIO(fileio.get(fileio.join_path(metas_path, file))) as f:
                meta = json.load(f)
                print(f"================detect dataset with traj {len(meta['datalist'])}==================")
                rng = np.random.default_rng(0)
                random.shuffle(meta['datalist'], rng.random)
                for data_path, len_data in meta['datalist'][:sample_num]: 
                    print(f"===load data {data_path}===")
                    with h5py.File(io.BytesIO(fileio.get(data_path)),'r') as file:
                        self.database[data_path] = {'observations/images/cam_high': file['observations/images/cam_high'][:],
                                                        'observations/images/cam_left_wrist': file['observations/images/cam_left_wrist'][:],
                                                        'observations/images/cam_right_wrist': file['observations/images/cam_right_wrist'][:],
                                                        'observations/qpos': file['observations/qpos'][:],
                                                        'observations/eef_6d': file['observations/eef_6d'][:]
                                                        }
                        self.datalist.extend([data_path, idx, meta['task_name']] for idx in range(len_data - 5))
                    
                
        self.image_aug = transforms.Compose([
            # transforms.RandomResizedCrop((224, 224), scale = (0.6, 1.0),ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        self.language_emb = torch.load("encoded_language.pt", map_location="cpu")
    
    def read_hdf5(self, datapath, idx, task_name):
        data = self.database[datapath]
        
        
        current_abs_joint = data['observations/qpos'][:]
        current_abs_eef6d = data['observations/eef_6d'][:]
        images = [data[key] for key in ['observations/images/cam_high', 'observations/images/cam_left_wrist', 'observations/images/cam_right_wrist']]
        image_input =  torch.stack([self.image_aug(decode_image_from_bytes(img[idx])) for img in images])
        query_duration = 2
        freq = 30
            
        time_stamp = np.arange(current_abs_joint.shape[0], dtype=np.float64) / float(freq)
        current_abs_joint = interp1d(
            time_stamp, 
            current_abs_joint,
            axis=0, 
            kind='linear', 
            bounds_error=False,  
            fill_value=(current_abs_joint[0], current_abs_joint[-1]), 
            assume_sorted=True
        )
        current_abs_eef6d = interp1d(
            time_stamp, 
            current_abs_eef6d,
            axis=0, 
            kind='linear', 
            bounds_error=False,  
            fill_value=(current_abs_eef6d[0], current_abs_eef6d[-1]), 
            assume_sorted=True
        )
        current_time = time_stamp[idx]
        query_time = np.linspace(current_time, 
                                 min(current_time + query_duration, max(time_stamp)), 
                                 self.num_actions+1).astype(np.float32)
        current_abs_joint = torch.tensor(current_abs_joint(query_time))
        current_abs_eef6d = torch.tensor(current_abs_eef6d(query_time))
        
        rel_joint = current_abs_joint[1:] - current_abs_joint[:1]
        rel_joint[:, 6] = current_abs_joint[1:, 6]
        rel_joint[:, 13] = current_abs_joint[1:, 13]
        
        rel_eef = current_abs_eef6d[1:] - current_abs_eef6d[:1]
        rel_eef[:, 9] = current_abs_eef6d[1:, 9]
        rel_eef[:, 19] = current_abs_eef6d[1:, 19]
        
        return {
            'encoded_language': self.language_emb[task_name],
            'images': image_input,
            
            'current_abs_joint': torch.tensor(current_abs_joint[0]).to(torch.float32),
            'current_abs_eef': torch.tensor(current_abs_eef6d[0]).to(torch.float32),
            
            'abs_joint_action':  torch.tensor(current_abs_joint[1:]).to(torch.float32),
            'abs_eef_action':  torch.tensor(current_abs_eef6d[1:]).to(torch.float32),
            
            'rel_joint_action': torch.tensor(rel_joint).to(torch.float32),
            'rel_eef_action': torch.tensor(rel_eef).to(torch.float32)
        }
        
    def read_airbot():
        pass
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        datapath, idx, task_name = self.datalist[index]
        return self.read_hdf5(datapath, idx, task_name)

def create_dataloader(
                 batch_size: int,
                 metas_path:str,
                 num_actions,
                 sample_num = 100000
                 ):
    dataset = MapstyleDataReader(
                 metas_path = metas_path,
                 num_actions = num_actions,
                 sample_num = sample_num
                 )
    # sampler = DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=rank) 
    train_dataloader = DataLoader(dataset, 
                                  batch_size=batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True,
                                  persistent_workers=True,
                                  drop_last=True)
    return train_dataloader