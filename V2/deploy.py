import model
import torch.nn as nn
from timm.models import create_model
from safetensors.torch import load_file
import io
from mmengine import fileio
import json_numpy
import argparse
import os
import json
import logging
import traceback
from typing import Any, Dict
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from scipy.spatial.transform import Rotation as R
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import glob
from scipy.spatial.transform import Rotation as R

def rotate6D_to_R(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats)

class DeployModel:
    def __init__(self, 
                 ckpt_path,
                 stats_path,
                 model_name = "model_base",
                 device = "cuda",
                 num_bins = 1
                ):
        self.device = device
        self.model_name = model_name
        self.model, self.lang_encoder = create_model(model_name)
        ckpt = load_file(ckpt_path)
        print(self.model.load_state_dict(ckpt, strict=False))
        self.model.to(torch.float32).to(self.device)
        # augmentations
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])
        print('Loading stats from', stats_path)
        try:
            stats = np.load(stats_path)
            self.global_mean = np.asarray(stats['mean'])
            self.global_std = np.asarray(stats['std'])
            print('global_mean', self.global_mean)
            print('global_std', self.global_std)
            try:
                self.global_min = np.asarray(stats['min'])
                self.global_max = np.asarray(stats['max'])
                print('global_min', self.global_mean)
                print('global_max', self.global_std)
            except Exception as e:  
                print('Error Loading stats', e)
            try:
                self.p5 = np.asarray(stats['p5'])
                self.p95 = np.asarray(stats['p95'])
                print('p5', self.p5)
                print('p95', self.p95)
            except Exception as e:  
                print('no p5 p95:', e)
        except Exception as e:
            print('Error Loading stats', e)
            self.global_mean = None
            self.global_std = None
            self.global_min = None
            self.global_max = None

        self.num_bins = num_bins

    def dequantize_action(self, quantized_action):
        quantized_action = np.asarray(quantized_action.cpu(), dtype=np.float32)
        try:
            print('normalize with p5 p95')            
            step = (self.p95 - self.p5) / (self.num_bins - 1)
            action = self.p5 + step * quantized_action
        except:
            print('no p5 p95')            
            step = (self.global_max - self.global_min) / (self.num_bins - 1)
            action = self.global_min + step * quantized_action
        # print('dequantized action:', action)
        return action

<<<<<<< HEAD
    def abs_recon(self, action_seq):
        # de-normalize 
        action_unnorm = action_seq.cpu() * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
        return action_unnorm

    def rel_recon(self, action_seq, proprio_start):
        # de-normalize 
        diff_denorm = action_seq.cpu() * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
        diff_denorm = np.asarray(diff_denorm).reshape(-1, 20)
        # recover absolute positions from diff
        # print('proprio_start', diff_denorm.shape, proprio_start.shape)
        left_xyz_start = proprio_start[:, :3]
        left_rot6_start = proprio_start[:, 3:9]
        left_grip_start = proprio_start[:, 10]
        right_xyz_start = proprio_start[:, 10:13]
        right_rot6_start = proprio_start[:, 13:19]
        right_grip_start = proprio_start[:, 19]
=======
    def proprio_norm(self, proprio):
        r = (proprio - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
        # print('proprio', proprio.shape) (14,) or (20,)
        return r.reshape(proprio.shape[0],)
>>>>>>> 10303c44ec3cfa69e91e86435de2df11703494e4

    def abs_recon(self, action_seq, proprio_start, discrete=False):
        print('un-normalizing for absoluate')
        if not discrete: # we don't do mean std normalize for discrete
            try:
                abs_denorm = action_seq * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
            except:
                abs_denorm = action_seq.cpu() * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
        else:
            abs_denorm = action_seq
        return abs_denorm

    def rel_recon(self, action_seq, proprio_start, discrete=False):
        print('proprio_start', proprio_start.shape)
        if not discrete: # we don't do mean std normalize for discrete
            try:
                diff_denorm = action_seq * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
            except:
                diff_denorm = action_seq.cpu() * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
        else:
            diff_denorm = action_seq
            print('Skipping mean std normalization since its discrete')
        if proprio_start.shape[-1] == 20:
            print('Processing relative ee')
            # de-normalize 
            diff_denorm = np.asarray(diff_denorm).reshape(-1, 20)
            # recover absolute positions from diff
            # print('proprio_start', diff_denorm.shape, proprio_start.shape)
            left_xyz_start = proprio_start[:, :3]
            left_rot6_start = proprio_start[:, 3:9]
            # left_grip_start = proprio_start[:, 10]
            right_xyz_start = proprio_start[:, 10:13]
            right_rot6_start = proprio_start[:, 13:19]
            # right_grip_start = proprio_start[:, 19]
            left_xyz_del = diff_denorm[:, :3]
            left_rot6_del = diff_denorm[:, 3:9]
            left_grip_del = diff_denorm[:, 9:10]
            right_xyz_del = diff_denorm[:, 10:13]
            right_rot6_del = diff_denorm[:, 13:19]
            right_grip_del = diff_denorm[:, 19:20]
            # print('left_xyz_del',left_xyz_del.shape)
            left_xyz = left_xyz_start + np.cumsum(left_xyz_del, axis=0)
            right_xyz = right_xyz_start + np.cumsum(right_xyz_del, axis=0)
            # print('left_xyz', left_xyz, left_xyz.shape)
            # left_rot6 = np.cumprod(left_rot6_del) * left_rot6_start
            left_rot6 = [left_rot6_start[0]]
            right_rot6 = [right_rot6_start[0]]
            for del_left, del_right in zip(left_rot6_del, right_rot6_del):
                left_rot6.append((rotate6D_to_R(del_left) * rotate6D_to_R(left_rot6[-1])).as_matrix()[:, :2].reshape(-1))
                right_rot6.append((rotate6D_to_R(del_right) * rotate6D_to_R(right_rot6[-1])).as_matrix()[:, :2].reshape(-1))
            left_rot6 = np.stack(left_rot6[1:])
            right_rot6 = np.stack(right_rot6[1:])
            # print('left_rot6', left_rot6, left_rot6.shape)
            left_grip = left_grip_del
            right_grip = right_grip_del
            # print('left_xyz', left_xyz.shape, left_rot6.shape)
            future_seq = np.concatenate([left_xyz, left_rot6, left_grip, right_xyz, right_rot6, right_grip], axis =-1)

        else:
            print('Processing relative qpos')
            diff_denorm = np.asarray(diff_denorm).reshape(-1, 14)
            left_qpos_start = proprio_start[:, :6]
            # left_grip_start = proprio_start[:, 6]
            right_qpos_start = proprio_start[:, 7:13]
            # right_grip_start = proprio_start[:, 12:13]

            left_qpos_del = diff_denorm[:, :6]
            left_grip_del = diff_denorm[:, 6:7]
            right_qpos_del = diff_denorm[:, 7:13]
            right_grip_del = diff_denorm[:, 13:14]

            print('left_qpos_del',left_qpos_del.shape, left_grip_del.shape)
            left_qpos = left_qpos_start + np.cumsum(left_qpos_del, axis=0)
            right_qpos = right_qpos_start + np.cumsum(right_qpos_del, axis=0)
            future_seq = np.concatenate([left_qpos, left_grip_del, right_qpos, right_grip_del], axis =-1)

        return future_seq

    def infer(self, payload: Dict[str, Any]):
        try:  
            self.model.eval()
            image_list = []        
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"]).astype(np.uint8)))
            language_inputs  = self.lang_encoder.encode_language(payload['language_instruction']).unsqueeze(0)
            
            # raw_bytes = base64.b64decode(payload["image0"])
            # img = Image.open(io.BytesIO(raw_bytes))
            image_input =  torch.stack([self.image_aug(img) for img in image_list])

            proprio = np.array(json_numpy.loads(payload['proprio']))
            proprio_normed = self.proprio_norm(proprio)
            # save lang
            print("proprio_normed:", proprio_normed.shape, proprio.shape)
            # print('current proprio', proprio)
            # print("payload['data_type']", payload['data_type'])

            # norm proprio
            # proprio_norm = (propio - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)

            inputs = {
                'encoded_language': torch.tensor(language_inputs).to(torch.float32).cuda(non_blocking=True),
                'images': torch.tensor(image_input).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
<<<<<<< HEAD
                'proprio':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True), # normalized proprio
=======
                'proprio':  torch.tensor(proprio_normed).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
>>>>>>> 10303c44ec3cfa69e91e86435de2df11703494e4
            }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs)
<<<<<<< HEAD
                print('action', action)
=======
                # print('action', action)
                discrete = False
                if 'dis' in self.model_name:
                        action = self.dequantize_action(action) # contain min max normalization (0, 1)
                        discrete = True
>>>>>>> 10303c44ec3cfa69e91e86435de2df11703494e4
                if 'data_type' in payload.keys():
                    if payload['data_type'] == 'rel':
                        # print('action', action.shape)
                        # print('proprio', proprio.shape)
                        action_sum = self.rel_recon(action, proprio[None, :], discrete=discrete) # contain mean std normalization (produces a normal distribution)
                        # print('action_sum', action_sum)
                        return JSONResponse(
                            {
                                'action': action.tolist(), 
                                'action_sum': action_sum.tolist(),
                                'global_mean': self.global_mean.tolist(),
                                'global_std': self.global_std.tolist()
                            }
                        )
<<<<<<< HEAD
                    else:
                        print('abs action', action.shape)
                        # action_unnorm = self.abs_recon(action)
=======
                    else: # we do mean std un-normalization for abs
                        action_unnorm = self.abs_recon(action, proprio[None, :], discrete=discrete) # contain mean std normalization (produces a normal distribution)
                        # print('action_sum', action_sum)
>>>>>>> 10303c44ec3cfa69e91e86435de2df11703494e4
                        return JSONResponse(
                            {
                                'action': action.tolist(), 
                                'action_unnorm': action_unnorm.tolist(),
                                'global_mean': self.global_mean.tolist(),
                                'global_std': self.global_std.tolist()
                            }
                        )
        
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            warning_str = "Your request threw an error; make sure your request complies with the expected Dict format"
            logging.warning(warning_str)
            return warning_str
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI()
        self.app.post("/act")(self.infer)
        uvicorn.run(self.app, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description='single-process evaluation on Calvin bench')
    parser.add_argument('--ckpt_path', type=str, help='load ckpt path')
    parser.add_argument('--model_name', default='model_base', type=str, help='load ckpt path')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    parser.add_argument("--stats_path", default='', type=str, help="Your global stats file for relative data / discrete models")

    
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)

    # print('stats_path', stats_path)
    # load your model
    if 'dis' in args.model_name:
        num_bins = 256
    else:
        num_bins = 1
    stats_path = None
    try:
        stats_paths = glob.glob(os.path.join(args.stats_path, "*.npz"))
        print('Availbale stats_paths:', stats_paths)
        for p in stats_paths:
            if 'ee' in args.model_name and 'ee' in p:
                if ('abs' in args.model_name and 'abs' in p) or \
                    ('rel' in args.model_name and 'rel' in p):
                    stats_path = p
                    break
            elif 'qpos' in args.model_name and 'qpos' in p:
                if ('abs' in args.model_name and 'abs' in p) or \
                    ('rel' in args.model_name and 'rel' in p):
                    stats_path = p
                    break
                
    except:
        stats_path = None
    print('FOUND stats_path:', stats_path)
    server = DeployModel(
        ckpt_path = ckpt_path,
        model_name = args.model_name,
        stats_path = stats_path,
        num_bins = num_bins
    )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()
