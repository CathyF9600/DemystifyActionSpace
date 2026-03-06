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
# from model import Normalizer

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
                 model_name = "model_base",
                 device = "cuda",
                 num_bins = 1,
                 norm_action = False
                ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.model_name = model_name
        self.norm_action = norm_action
        self.model, self.lang_encoder = create_model(model_name, num_views = 3)
        ckpt = load_file(ckpt_path)
        # print(ckpt.keys())
        print(self.model.load_state_dict(ckpt, strict=True))
        self.model.to(torch.float32).to(self.device)
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])


    def infer(self, payload: Dict[str, Any]):
        try:  
            self.model.eval()
            image_list = []        
            # if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"]).astype(np.uint8)))
            # if "image1" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image1"]).astype(np.uint8))) 
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"])))
            if "image1" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image1"])))
            if "image2" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image2"])))
            language_inputs  = self.lang_encoder.encode_language(payload['language_instruction']).unsqueeze(0)
            
            # raw_bytes = base64.b64decode(payload["image0"])
            # img = Image.open(io.BytesIO(raw_bytes))
            # print('len(image_list)', len(image_list))
            image_input =  torch.stack([self.image_aug(img) for img in image_list])

            proprio = np.array(json_numpy.loads(payload['proprio']))
            inputs = {
                'encoded_language': torch.tensor(language_inputs).to(torch.float32).cuda(non_blocking=True),
                'images': torch.tensor(image_input).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                'current_abs_joint':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                'current_abs_eef': torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True)
            }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs)
                # print(action[:, 9], proprio[9])
                if self.model.control_interface == 'rel_joint' or self.model.control_interface == 'rel_eef':
                    # print('action', action.shape)
                    # print('proprio', proprio.shape)
                    # action_sum = self.rel_recon(action, proprio[None, :])[None,:] # contain mean std normalization (produces a normal distribution)
                    print('model.control_interface', self.model.control_interface, self.ckpt_path)
                    action_sum = action.cpu().numpy() + proprio[None, :] # (1, 30, 20)   +   (1, 20) = (1, 30, 20)   +   (1, 1, 20)   ->   (1, 30, 20) broadcasting
                    if self.model.control_interface == 'rel_joint': # gripper dim is 6 and 13
                        action_sum[:, :, 6] = action[:, :, 6].cpu().numpy()
                        action_sum[:, :, 13] = action[:, :, 13].cpu().numpy()
                    elif self.model.control_interface == 'rel_eef': # gripper dim is 9 and 19
                        action_sum[:, :, 9] = action[:,:, 9].cpu().numpy()
                        action_sum[:, :, 19] = action[:, :, 19].cpu().numpy()
                        
                    print('action_sum', action_sum.shape)
                    return JSONResponse(
                        {
                            'action': action_sum.tolist()
                        }
                    )
                else:
                    # print('abs action', action.shape)
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
    parser.add_argument('--norm_action', default=False, action='store_true', help='load ckpt path')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    # parser.add_argument("--stats_path", default='', type=str, help="Your global stats file for relative data / discrete models")

    
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)

    stats_path = None
    server = DeployModel(
        ckpt_path = ckpt_path,
        model_name = args.model_name,
        num_bins = 1,
        norm_action = args.norm_action
        )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()
