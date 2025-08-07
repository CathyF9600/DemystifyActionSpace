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
import model

class DeployModel:
    def __init__(self, 
                 ckpt_path,
                 model_name = "model_base",
                 device = "cuda",
                ):
        self.device = device
        self.model, self.lang_encoder = create_model(model_name)
        print(self.model.load_state_dict(load_file(ckpt_path), strict=False))
        self.model.to(torch.float32).to(self.device)
        # augmentations
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)
        ])

    def infer(self, payload: Dict[str, Any]):
        try:  
            self.model.eval()
            image_list = []        
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"])))
            language_inputs  = self.lang_encoder.encode_language(payload['language_instruction']).unsuqeeze(0)
            image_input =  torch.stack([self.image_aug(img) for img in image_list])
            proprio = np.array(json_numpy.loads(payload['proprio']))
            # save lang
            print("language:", payload['language_instruction'])
            print('current proprio', proprio)
            inputs = {
                'encoded_language': torch.tensor(language_inputs).to(torch.float32).cuda(non_blocking=True),
                'images': torch.tensor(image_input).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                'proprio':  torch.tensor(proprio).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
            }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs)
                print(action)
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
    parser.add_argument('--ckpt_path', type=str, help='load ckpt path')
    parser.add_argument('--model_name', default='model_base', type=str, help='load ckpt path')
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")

        
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)
    # load your model
    server = DeployModel(
        ckpt_path = args.ckpt_path,
        model_name = args.model_name
    )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()
