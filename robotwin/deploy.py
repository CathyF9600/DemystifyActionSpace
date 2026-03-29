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

def quat_to_rotate6D(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6D(euler: np.ndarray) -> np.ndarray:
    """
    euler: (..., 3)  # xyz order, radians
    return: (..., 6)
    """
    rotmat = R.from_euler('xyz', euler).as_matrix()  # (..., 3, 3)
    return rotmat[..., :, :2].reshape(euler.shape[:-1] + (6,))

class DeployModel:
    def __init__(self, 
                 ckpt_path,
                 model_name = "model_base",
                 device = "cuda",
                 num_bins = 1,
                 norm_action = False
                ):
        self.device = device
        self.model_name = model_name
        self.norm_action = norm_action
        self.model, self.lang_encoder = create_model(model_name)
        ckpt = load_file(ckpt_path)
        print(self.model.load_state_dict(ckpt, strict=False))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(torch.float32).to(self.device)
        # augmentations
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

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

    def proprio_norm(self, proprio):
        print('norming proprio')
        r = (proprio - self.global_mean[None, :]) / (self.global_std[None, :] + 1e-8)
        # print('proprio', proprio.shape) (14,) or (20,)
        return r.reshape(proprio.shape[0],)

    def abs_recon(self, action_seq, proprio_start, discrete=False):
        if not discrete: # we don't do mean std normalize for discrete
            print('denormalizing for absoluate')
            try:
                abs_denorm = action_seq * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
            except:
                abs_denorm = action_seq.cpu() * (self.global_std[None, :] + 1e-8) + self.global_mean[None, :]
        else:
            abs_denorm = action_seq
        return abs_denorm

    def rel_recon(self, action, discrete=False, chunk_wise=True, rot_repr=None):
        print('action', action.shape)
        if rot_repr != None: # for ee, rot_repr exists
            print('Processing relative ee')
            if rot_repr == "rot6d":
                action_sum[:, :, 9] = action[:,:, 9]
                action_sum[:, :, 19] = action[:, :, 19]
            elif rot_repr == "quat":
                action_sum[:, :, 7] = action[:,:, 7]
                action_sum[:, :, 15] = action[:, :, 15]
                action_sum = quat_to_rotate6D(action_sum)
            elif rot_repr == "euler":
                action_sum[:, :, 6] = action[:,:, 6]
                action_sum[:, :, 13] = action[:, :, 13]
                action_sum = euler_to_rotate6D(action_sum)
        else: # joint
            # gripper dim is 6 and 13
            action_sum[:, :, 6] = action[:, :, 6]
            action_sum[:, :, 13] = action[:, :, 13]
        return action_sum.squeeze()

    def infer(self, payload: Dict[str, Any]):
        print('norm_action', self.norm_action)
        try:  
            self.model.eval()
            image_list = []        
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"]).astype(np.uint8)))
            language_inputs  = self.lang_encoder.encode_language(payload['language_instruction']).unsqueeze(0)
            
            # raw_bytes = base64.b64decode(payload["image0"])
            # img = Image.open(io.BytesIO(raw_bytes))
            image_input =  torch.stack([self.image_aug(img) for img in image_list])

            proprio = np.array(json_numpy.loads(payload['proprio']))
            print('client proprio', proprio)
            if 'rel' in self.model_name:
                inputs = {
                    'encoded_language': torch.tensor(language_inputs).to(torch.float32).cuda(non_blocking=True),
                    'images': torch.tensor(image_input).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                    'proprio':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True), # normalized proprio
                    'chunk_wise_delta': True,
                    # 'rot_repr': 'rot6d'
            }
            else:
                inputs = {
                    'encoded_language': torch.tensor(language_inputs).to(torch.float32).cuda(non_blocking=True),
                    'images': torch.tensor(image_input).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                    'proprio':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True), # normalized proprio
                }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs)
                discrete = False
                if 'dis' in self.model_name:
                        action = self.dequantize_action(action) # contain min max normalization (0, 1)
                        discrete = True
                if 'rel' in self.model_name:
                    print('action', action.shape)
                    return JSONResponse(
                        {
                            'action_sum': action.tolist(), 
                        }
                    )
                else:
                    print('abs action', action.shape)
                    return JSONResponse(
                        {
                            'action': action.tolist()
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
    parser.add_argument('--norm_action', default=False, type=bool, help='load ckpt path')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    parser.add_argument("--stats_path", default='', type=str, help="Your global stats file for relative data / discrete models")

    
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)

    server = DeployModel(
        ckpt_path = ckpt_path,
        model_name = args.model_name,
        num_bins = 1,
        norm_action = args.norm_action
    )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()
