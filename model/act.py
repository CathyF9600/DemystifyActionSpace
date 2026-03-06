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


def get_positional_embeddings(seq_length, d_model):
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


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
    def __init__(self, meta_path = "assets/encoded_language.pt"):
        self.language_emb = torch.load(meta_path, map_location="cpu")
        print(f"successfully load language hub: {self.language_emb.keys()}")
        
    @torch.no_grad()
    def encode_language(self, language_inputs: str):
        return self.language_emb[language_inputs]

class ACT(nn.Module):
    def __init__(self,
                 depth = 3,
                 hidden_size = 512,
                 vision_backbone = "FiLM_Resnet18",
                 control_interface = "abs",
                 dim_language = 768,
                 dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                 dim_actions = 20, # 14 for euler angles
                 num_action_chunk = 30,
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
        self.proprio_proj = nn.Linear(dim_proprio, hidden_size)
        self.language_proj = nn.Linear(dim_language, hidden_size)
        
        self.queries = nn.Parameter(torch.zeros(1, num_action_chunk, hidden_size), requires_grad=True)
        self.queries_pos_emb = nn.Parameter(get_positional_embeddings(num_action_chunk, hidden_size), requires_grad=False)
        self.input_pos_emb = nn.Parameter(get_positional_embeddings(100, hidden_size), requires_grad=False)
        
        self.decoder = nn.Transformer(
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers = 4,
            num_decoder_layers = 2,
            dim_feedforward = hidden_size * 4,
            dropout = 0.1,
            batch_first = True,
            norm_first = False,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, dim_actions)
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
        vision_embedding = vision_embedding.permute(0, 2, 1).reshape(B, V*N, num_features)
        input_x = torch.cat(
            [vision_embedding, self.language_proj(encoded_language).view(B, 1, -1), self.proprio_proj(proprio).unsqueeze(1)], dim = 1
        ) + self.input_pos_emb

        queries = self.queries.repeat(B, 1, 1) + self.queries_pos_emb
        queries = self.decoder(input_x, queries)
        queries = self.norm(queries)
        output_action = self.head(queries)
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
        
        # print('xxxxxxxxxxxxxxxxxx image', images.shape, encoded_language.shape, proprio.shape)
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone(images.view(B*V, C, H, W), encoded_language.view(B, 1, -1).repeat(1, V, 1).view(B*V, -1)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).reshape(B, V*N, num_features)
        input_x = torch.cat(
            [vision_embedding, self.language_proj(encoded_language).view(B, 1, -1), self.proprio_proj(proprio).unsqueeze(1)], dim = 1
        ) + self.input_pos_emb

        queries = self.queries.repeat(B, 1, 1) + self.queries_pos_emb
        queries = self.decoder(input_x, queries)
        queries = self.norm(queries)
        output_action = self.head(queries)
        
        if 'rel' in self.control_interface: output_action /= self.delta_action_scale
        return output_action



@register_model
def model_abs_ee_act(
                **kwargs):
    model = ACT(
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
def model_rel_ee_act(
                **kwargs):
    model = ACT(
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
def model_rel_joint_act(
                **kwargs):
    model = ACT(
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
def model_abs_joint_act(
                **kwargs):
    model = ACT(
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