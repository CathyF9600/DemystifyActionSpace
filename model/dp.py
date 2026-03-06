import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models import create_model
import timm.models
from typing import List
from transformers import AutoTokenizer, SiglipTextModel
from timm.models.vision_transformer import Mlp
import math
import json
import model.backbone



def basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)    

class TimeEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))
    
class language_encoder:
    def __init__(self, meta_path = "encoded_language.pt"):
        self.language_emb = torch.load(meta_path, map_location="cpu")
        print(f"successfully load language hub: {self.language_emb.keys()}")
        
    @torch.no_grad()
    def encode_language(self, language_inputs: str):
        return self.language_emb[language_inputs]

class MlpDecoder(nn.Module):
    def __init__(self, 
                 depth = 2,
                 hidden_size = 512,
                 mlp_ratio = 4.0,
                 dim_visual = 512,
                 num_views = 1,
                 dim_actions = 20,
                 dim_proprio = 20,
                 num_action_chunk = 5
                 ):
        super().__init__()
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        self.time_encoder = TimeEmbedder(hidden_size // 4)
        in_dim = dim_visual * num_views + dim_proprio \
                + dim_actions * num_action_chunk + hidden_size // 4
        self.in_proj = Mlp(in_features=in_dim, 
                            hidden_features=int(hidden_size * mlp_ratio), 
                            out_features=hidden_size,
                            act_layer=lambda: nn.GELU(approximate="tanh"), 
                        drop=0.1)


        self.blocks = nn.ModuleList([Mlp(in_features=hidden_size, 
                                        hidden_features=int(hidden_size * mlp_ratio), 
                                        act_layer=lambda: nn.GELU(approximate="tanh"), 
                                    drop=0.1) for _ in range(depth)])
        
        self.ln = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(depth)])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, dim_actions * num_action_chunk)
        )
        self.apply(basic_init)
    

    
    def forward(self, 
                visual_feature, # B num_views num_tokens dim_visual
                proprio, # B dim_proprio
                noise_action = None, # B num_action_chunk dim_action
                t = None # B
        ):
        batch_size = visual_feature.shape[0]
        visual_feature = torch.mean(visual_feature, dim=-2, keepdim=False)
        x = torch.cat([visual_feature.view(batch_size, -1), 
                                proprio], dim=-1)
        x = torch.cat([x, noise_action.view(batch_size, -1), self.time_encoder(t)], dim =-1)            
        x = self.in_proj(x)
        for block, ln in zip(self.blocks, self.ln): x = x + block(ln(x))
        return self.out_proj(x).view(batch_size, self.num_action_chunk, -1)


class DP(nn.Module):
    def __init__(self,
                 depth = 3,
                 vision_backbone = "FiLM_Resnet18",
                 control_interface = "abs_joint",
                 dim_language = 768,
                 dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                 dim_actions = 20, # 14 for euler angles
                 num_action_chunk = 10,
                 loss_scale = 10,
                 delta_action_scale = 10,
                 **kwargs
                 ):
        super().__init__()
        self.delta_action_scale = delta_action_scale
        self.loss_scale = loss_scale
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        self.control_interface = control_interface
        self.vision_backbone = create_model(vision_backbone)
        self.decoder = MlpDecoder(
                                    depth = depth,
                                    hidden_size = 512,
                                    mlp_ratio = 4.0,
                                    dim_visual = 512,
                                    num_views = 2,
                                    dim_actions = dim_actions,
                                    dim_proprio = dim_proprio,
                                    num_action_chunk = num_action_chunk,
                                    )
        self.loss = nn.MSELoss()

    def forward(self,
                images: torch.FloatTensor, # B * V * C * H * W,
                encoded_language: torch.Tensor, # B C
                
                current_abs_joint: torch.Tensor, # B C
                current_abs_eef: torch.Tensor, # B C
                
                abs_joint_action: torch.Tensor, # B C
                abs_eef_action: torch.Tensor, # B C
                
                rel_joint_action: torch.Tensor,
                rel_eef_action: torch.Tensor):
        self.train()
        if self.control_interface == "abs_joint":
            action_seq = abs_joint_action
            proprio = current_abs_joint
        elif self.control_interface == "rel_joint":
            action_seq = rel_joint_action * self.delta_action_scale
            proprio = current_abs_joint
        elif self.control_interface == "abs_eef":
            action_seq = abs_eef_action
            proprio = current_abs_eef
        elif self.control_interface == "rel_eef":
            action_seq = rel_eef_action * self.delta_action_scale
            proprio = current_abs_eef
        
        
    
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone(images.view(B*V, C, H, W), encoded_language.view(B, 1, -1).repeat(1, V, 1).view(B*V, -1)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        # Flow Matching
        t = (torch.rand(1, device=images.device) + torch.arange(images.shape[0], device=images.device) / images.shape[0]) % (1 - 1e-5)
        noise = torch.randn_like(action_seq)
        noise_action = noise * t.view(-1, 1, 1) + action_seq * (1 - t).view(-1, 1, 1)
        output_action = self.decoder(
            visual_feature = vision_embedding,
            proprio = proprio,
            noise_action = noise_action, # B num_action_chunk dim_action
            t = t # B
        )
        return self.loss(output_action, action_seq) * self.loss_scale
        
        
    def pred_action(self,
                images: torch.Tensor, # B V C H W
                encoded_language: torch.Tensor, # B C
                current_abs_joint: torch.Tensor, # B C
                current_abs_eef: torch.Tensor, # B C
                steps = 10,
                **kwargs
            ):
        self.eval()
        if self.control_interface == "abs_joint":
            proprio = current_abs_joint
        elif self.control_interface == "rel_joint":
            proprio = current_abs_joint
        elif self.control_interface == "abs_eef":
            proprio = current_abs_eef
        elif self.control_interface == "rel_eef":
            proprio = current_abs_eef

        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone(images.view(B*V, C, H, W), encoded_language.view(B, 1, -1).repeat(1, V, 1).view(B*V, -1)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)

        action_with_noise = torch.randn(B, self.num_action_chunk, self.dim_actions,
                                        device = images.device)
        for i in range(steps, 0, -1):
            time = torch.full((B,), i / steps, device=images.device)
            pred_action = self.decoder(      
                        visual_feature = vision_embedding,
                        proprio = proprio,
                        noise_action = action_with_noise, # B num_action_chunk dim_action
                        t = time)
            action_with_noise = action_with_noise - (action_with_noise - pred_action) / time.view(B, 1, 1) / steps
        
        if 'rel' in self.control_interface: action_with_noise /= self.delta_action_scale
        
        return action_with_noise


@register_model
def model_abs_ee_flow(
                **kwargs):
    model = DP(
        depth = 3,
        vision_backbone = "FiLM_Resnet18",
        control_interface = "abs_eef",
        dim_language = 768,
        dim_proprio = 20, # 14 for euler angles, 20 for rot6d
        dim_actions = 20, # 14 for euler angles
        num_action_chunk = 30,
        loss_scale = 10
    )
    return model, language_encoder()


@register_model
def model_rel_ee_flow(
                **kwargs):
    model = DP(
        depth = 3,
        vision_backbone = "FiLM_Resnet18",
        control_interface = "rel_eef",
        dim_language = 768,
        dim_proprio = 20, # 14 for euler angles, 20 for rot6d
        dim_actions = 20, # 14 for euler angles
        num_action_chunk = 30,
        loss_scale = 10
    )
    return model, language_encoder()


@register_model
def model_rel_joint_flow(
                **kwargs):
    model = DP(
        depth = 3,
        vision_backbone = "FiLM_Resnet18",
        control_interface = "rel_joint",
        dim_language = 768,
        dim_proprio = 14, # 14 for euler angles, 20 for rot6d
        dim_actions = 14, # 14 for euler angles
        num_action_chunk = 30,
        loss_scale = 10
    )
    return model, language_encoder()

@register_model
def model_abs_joint_flow(
                **kwargs):
    model = DP(
        depth = 3,
        vision_backbone = "FiLM_Resnet18",
        control_interface = "abs_joint",
        dim_language = 768,
        dim_proprio = 14, # 14 for euler angles, 20 for rot6d
        dim_actions = 14, # 14 for euler angles
        num_action_chunk = 30,
        loss_scale = 10,
        delta_action_scale = 10
    )
    return model, language_encoder()