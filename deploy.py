import model
import torch.nn as nn
from timm.models import create_model
from model import BaseModel, language_encoder
from safetensors.torch import load_file
import io
from mmengine import fileio
import json_numpy
import argparse
import os
json_numpy.patch()
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
import cv2
from io import BytesIO

def decode_image_from_bytes(camera_rgb_image):
    # print('camera_rgb_image', type(camera_rgb_image))
    # camera_rgb_image = np.array(camera_rgb_image)
    if isinstance(camera_rgb_image, Image.Image):
        buffer = BytesIO()
        camera_rgb_image.save(buffer, format="PNG")  # Or "JPEG"
        camera_rgb_image = buffer.getvalue()
    if isinstance(camera_rgb_image, (bytes, bytearray)): camera_rgb_image = np.frombuffer(camera_rgb_image, dtype=np.uint8)
    rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
    if rgb is None: 
        rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8) 
        if rgb.size == 2764800: 
            rgb = rgb.reshape(720, 1280, 3) 
        elif rgb.size == 921600: 
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)

class DeployModel:
    def __init__(self, 
                 ckpt_path,
                 meta_files,
                 model = None,
                 model_type = "flow-matching",
                 device = "cuda",
                 denoising_steps = 5,
                ):
        self.device = device
        self.model = model
        # Initialize language encoder (frozen)
        self.lang_encoder = language_encoder()
        self.lang_encoder.eval()  # No training for language encoder

        print(self.model.load_state_dict(load_file(ckpt_path), strict=False))
        self.model.to(torch.float32).to(self.device)
        with io.BytesIO(fileio.get(meta_files)) as f:
            self.meta = json.load(f)
        self.denoising_steps = denoising_steps
        
        # augmentations
        self.image_aug = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale = (0.8, 1.0),ratio=(1.0, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

    def infer(self, payload: Dict[str, Any]):
        try:  
            self.model.eval()
            
            image_list = []        
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"])))
            if "image1" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image1"])))
            if "image2" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image2"])))
            language_inputs  = self.lang_encoder.forward([payload['language_instruction']])
            # image_inputs = self.image_processor(image_list)
            image_input =  torch.stack([self.image_aug(img) for img in image_list])

            proprio = np.array(json_numpy.loads(payload['proprio']))
            print('language_inputs',language_inputs)
            print('image_inputs',image_inputs)
            inputs = {
                **{key: value.cuda(non_blocking=True) for key, value in language_inputs.items()},
                # **{key: value.cuda(non_blocking=True) for key, value in image_inputs.items()},
                'images': image_input,
                'proprio':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                'hetero_info': torch.tensor(self.meta['domain_id']).unsqueeze(0).cuda(non_blocking=True),
                'steps': self.denoising_steps
            }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs).squeeze(0).cpu().numpy()
                if "data_type" in payload.keys():
                    if payload["data_type"] == "rel":
                        action_final = action.cumsum(axis = 1) + proprio
            return JSONResponse(
                {'action': action.tolist()})
        
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
    parser.add_argument('--ckpt_path', default='/data2/UniActV2/Calvin-Rel/140K', type=str, help='load ckpt path')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    parser.add_argument('--meta_files', default='/data2/UniActV2/Calvin-Rel/140K/Calvin_Rel.json', type=str, help='load meta files')
    parser.add_argument("--denoising_steps", default=5, type=int, help="denosing steps for diffusion model")

    # Model parameters
    parser.add_argument('--vision_backbone', default="resnet18.a1_in1k", type=str, help="Vision backbone name (from timm)")
    parser.add_argument('--decoder_name', default="mlp_decoder_base", type=str, help="Decoder name")
    parser.add_argument('--model_type', type=str, default="continuous", choices=["continuous", "discrete", "flow-matching"], help="Model type")
    parser.add_argument('--num_actions', type=int, default=10, help="Number of action chunks")
    parser.add_argument('--learning_coef', default=1., type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)

    # Resume & Checkpoint Save & evaluation parameters
    parser.add_argument('--save_interval', default=20000, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--dim_actions', default=20, type=int)
    parser.add_argument('--dim_proprio', default=20, type=int)
        
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)
    model = BaseModel(
        vision_backbone=args.vision_backbone,
        model_type=args.model_type,
        decoder_name=args.decoder_name,
        num_action_chunk=args.num_actions,
        dim_actions=args.dim_actions,  # Matches dataset's action dimension
        dim_proprio=args.dim_proprio
    )
    # load your model
    server = DeployModel(
        ckpt_path = ckpt_path,
        meta_files=kwargs['meta_files'],
        model = model,
        model_type = kwargs['model_type'],
        device = torch.device("cuda"),
        denoising_steps= kwargs['denoising_steps'],
    )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()
