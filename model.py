import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models import create_model
import timm.models
from typing import List
from transformers import AutoTokenizer, SiglipTextModel
import decoders

class language_encoder(nn.Module):
    def __init__(self, model_name = "google/siglip-base-patch16-224"):
        super().__init__()
        self.model = SiglipTextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    @torch.no_grad()
    def forward(self, language_inputs: List[str]):
        inputs = self.tokenizer(language_inputs, padding="max_length", return_tensors="pt")        
        # 2. 将输入张量移动到模型所在的设备（关键修复）
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        return self.model(**inputs).pooler_output

class BaseModel(nn.Module):
    def __init__(self,
                 vision_backbone = "resnet18.a1_in1k",
                 model_type = "continuous",
                 decoder_name = "mlp_decoder_base",
                 dim_language = 768,
                 dim_proprio = 20, # 14 for euler angles
                 dim_actions = 20, # 14 for euler angles
                 num_action_chunk = 10,
                 num_bins = 256, ## for discrete policy only
                 **kwargs
                 ):
        super().__init__()
        self.model_type = model_type
        self.num_bins = num_bins
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        assert model_type in ['continuous', 'discrete', 'flow-matching']
        self.vision_backbone = create_model(vision_backbone, pretrained=True)
        del self.vision_backbone.fc
        self.decoder = create_model(decoder_name,
                                    model_type = model_type,
                                    dim_visual = self.vision_backbone.num_features,
                                    dim_language = dim_language,
                                    dim_proprio = dim_proprio,
                                    dim_actions = dim_actions,
                                    num_action_chunk = num_action_chunk,
                                    num_bins = num_bins)
        
        if model_type == 'discrete': self.loss = nn.CrossEntropyLoss()
        else: self.loss = nn.MSELoss()
        

    def forward(self,
                images: torch.FloatTensor, # B * V * C * H * W,
                encoded_language: torch.Tensor, # B C
                abs_eef: torch.Tensor):
        # print('abs_eef', abs_eef.shape)
        actions = abs_eef[:, 1:] # 0~19: abs future eef
        proprio = abs_eef[:, 0] + torch.randn_like(abs_eef[:, 0]) * 0.05 # augmentation
        
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        noise_action = None
        t = None
        if self.model_type == 'flow-matching':
            t = (torch.rand(1, device=images.device) + torch.arange(images.shape[0], device=images.device) / images.shape[0]) % (1 - 1e-5)
            noise = torch.randn_like(actions)
            noise_action = noise * t.view(-1, 1, 1) + actions * (1 - t).view(-1, 1, 1)
        # print('******', vision_embedding.shape, encoded_language.shape)
        # print(proprio.shape, noise_action.shape, t.shape)

        output_action = self.decoder(
            visual_feature = vision_embedding,
            language_feature = encoded_language,
            proprio = proprio,
            noise_action = noise_action, # B num_action_chunk dim_action
            t = t # B
        )
        
        if self.model_type == 'flow-matching': 
            # print('output_action', output_action.shape, actions.shape)
            return self.loss(output_action, noise - actions)
        elif self.model_type == 'discrete':
            return self.loss(output_action.view(-1, self.num_bins), actions.view(-1))
        else:
            return self.loss(output_action, actions)
        
        
    def pred_action(self,
                images: torch.Tensor, # B V C H W
                encoded_language: torch.Tensor, # B C
                proprio: torch.Tensor,
                steps = 5, # for flow-matching only
            ):
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        
        
        if self.model_type == 'flow-matching': # DP
            action_with_noise = torch.randn(B, self.num_action_chunk, self.dim_actions,
                                            device = images.device)
            for i in range(steps, 0, -1):
                time = torch.full((B,), i / steps, device=images.device)
                pred_action = self.decoder(      
                            visual_feature = vision_embedding,
                            language_feature = encoded_language,
                            proprio = proprio,
                            noise_action = action_with_noise, # B num_action_chunk dim_action
                            t = time)
                action_with_noise = action_with_noise - pred_action / time.view(B, 1, 1) / steps
        elif self.model_type == 'discrete': # Auto-regressive model
            pred_action = self.decoder(      
                    visual_feature = vision_embedding,
                    language_feature = encoded_language,
                    proprio = proprio).view(B, self.num_action_chunk, self.dim_actions, self.num_bins).argmax(dim=-1)
            
        elif self.model_type == 'continuous': # ACT
            pred_action = self.decoder(      
                    visual_feature = vision_embedding,
                    language_feature = encoded_language,
                    proprio = proprio)
        return pred_action

    


