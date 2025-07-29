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

class DeployModel:
    def __init__(self, 
                 ckpt_path,
                 meta_files,
                 model_type = "flow-matching",
                 device = "cuda",
                 denoising_steps = 5,
                 **kwargs):
        self.device = device
        # Initialize language encoder (frozen)
        lang_encoder = language_encoder()
        lang_encoder.eval()  # No training for language encoder
        
        model = BaseModel(
            vision_backbone=args.vision_backbone,
            model_type=args.model_type,
            decoder_name=args.decoder_name,
            num_action_chunk=args.num_actions,
            dim_actions=args.dim_actions,  # Matches dataset's action dimension
            dim_proprio=args.dim_proprio
        )
        print(self.model.load_state_dict(load_file(ckpt_path), strict=False))
        self.model.to(torch.float32).to(self.device)
        with io.BytesIO(fileio.get(meta_files)) as f:
            self.meta = json.load(f)
        self.denoising_steps = denoising_steps
            
    def infer(self, payload: Dict[str, Any]):
        try:  
            self.model.eval()
            
            image_list = []        
            if "image0" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image0"])))
            if "image1" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image1"])))
            if "image2" in payload.keys(): image_list.append(Image.fromarray(json_numpy.loads(payload["image2"])))
            language_inputs  = self.text_processor.encode_language([payload['language_instruction']])
            image_inputs = self.image_processor(image_list)
            action = np.array(json_numpy.loads(payload['action']))
            inputs = {
                **{key: value.cuda(non_blocking=True) for key, value in language_inputs.items()},
                **{key: value.cuda(non_blocking=True) for key, value in image_inputs.items()},
                'proprio':  torch.tensor(action).to(torch.float32).unsqueeze(0).cuda(non_blocking=True),
                'hetero_info': torch.tensor(self.meta['domain_id']).unsqueeze(0).cuda(non_blocking=True),
                'steps': self.denoising_steps
            }
            
            with torch.no_grad():
                action = self.model.pred_action(**inputs).squeeze(0).cpu().numpy()
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
    parser.add_argument('--model_type', default='flow-matching', type=str, help='create model name')
    parser.add_argument('--vision_backbone', default="resnet18.a1_in1k", type=str, help="Vision backbone name (from timm)")
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    parser.add_argument('--meta_files', default='/data2/UniActV2/Calvin-Rel/140K/Calvin_Rel.json', type=str, help='load meta files')
    parser.add_argument("--denoising_steps", default=5, type=int, help="denosing steps for diffusion model")
    parser.add_argument("--dim_actions", default=7, type=int, help="denosing steps for diffusion model")
    args = parser.parse_args()
    kwargs = vars(args)
    ckpt_path = os.path.join(kwargs['ckpt_path'], 'model.safetensors')
    print("-"*88)
    print('ckpt path:', ckpt_path)
    print("-"*88)
    
    # load your model
    server = DeployModel(
        ckpt_path = ckpt_path,
        meta_files=kwargs['meta_files'],
        model_type = kwargs['model_type'],
        device = torch.device("cuda"),
        denoising_steps= kwargs['denoising_steps'],
        dim_actions = kwargs['dim_actions']
    )  
    server.run(host=kwargs['host'], port=kwargs['port'])
    
if __name__ == '__main__':
    main()