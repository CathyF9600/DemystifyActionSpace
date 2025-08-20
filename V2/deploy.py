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
                ):
        self.device = device
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
            self.global_min = np.asarray(stats['min'])
            self.global_max = np.asarray(stats['max'])
            print('global_mean', self.global_mean)
            print('global_std', self.global_std)
        except:
            print('stats_path is empty')
    # def dequantize_action():

    
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
        prorpio_recon = np.concatenate([
            proprio_start,
            future_seq
        ], axis=0)
        # print("prorpio_recon: ", prorpio_recon)
        return prorpio_recon

    def infer(self, payload: Dict[str, Any]):
        try:  
            self.model.eval()
            image_list = []        
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"])))
            language_inputs  = self.lang_encoder.encode_language(payload['language_instruction']).unsqueeze(0)
            image_input =  torch.stack([self.image_aug(img) for img in image_list])
            proprio = np.array(json_numpy.loads(payload['proprio']))
            # save lang
            # print("language:", payload['language_instruction'])
            # print('current proprio', proprio)
            # print("payload['data_type']", payload['data_type'])

            inputs = {
                'encoded_language': torch.tensor(language_inputs).to(torch.float32).cuda(non_blocking=True),
                'images': torch.tensor(image_input).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                'proprio':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
            }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs)
                # print('action', action)
                if 'data_type' in payload.keys():
                    if payload['data_type'] == 'rel':
                        # print('action', action.shape)
                        # print('proprio', proprio.shape)
                        action_sum = self.rel_recon(action, proprio[None, :])
                        # print('action_sum', action_sum)
                        return JSONResponse(
                            {
                                'action': action.tolist(), 
                                'action_sum': action_sum.tolist(),
                                'global_mean': self.global_mean.tolist(),
                                'global_std': self.global_std.tolist()
                            }
                        )
                    else:
                        return JSONResponse(
                            {'action': action.tolist(), }
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
    # parser.add_argument("--stats_path", default='', type=str, help="Your global stats file for relative data / discrete models")

    
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)
    try:
        stats_path = glob.glob(os.path.join(args.ckpt_path, "*.npz"))[0]
    except:
        stats_path = None
    # print('stats_path', stats_path)
    # load your model
    server = DeployModel(
        ckpt_path = ckpt_path,
        model_name = args.model_name,
        stats_path = stats_path
    )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()
