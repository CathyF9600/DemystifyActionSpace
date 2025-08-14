import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models import create_model
import timm.models
from typing import List
from transformers import AutoTokenizer, SiglipTextModel
from timm.models.vision_transformer import Mlp


def basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)    

class language_encoder:
    def __init__(self, meta_path = "encoded_language.pt"):
        self.language_emb = torch.load(meta_path, map_location="cuda")
        
    @torch.no_grad()
    def encode_language(self, language_inputs: str):
        return self.language_emb[language_inputs]

class MlpDecoder(nn.Module):
    def __init__(self, 
                 depth = 2,
                 hidden_size = 512,
                 mlp_ratio = 4.0,
                 dim_visual = 512,
                 dim_language = 768,
                 num_views = 1,
                 dim_actions = 20,
                 dim_proprio = 20,
                 num_action_chunk = 5,
                 num_bins = 256
                 ):
        super().__init__()
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        self.in_proj = Mlp(in_features=dim_visual * num_views + dim_language + dim_proprio, 
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
                nn.Linear(hidden_size, dim_actions * num_action_chunk * num_bins)
        )
        self.apply(basic_init)
    
    def forward(self, 
                visual_feature, # B num_views num_tokens dim_visual
                language_feature, # B dim_language
                proprio, # B dim_proprio
        ):
        batch_size = visual_feature.shape[0]
        visual_feature = torch.mean(visual_feature, dim=-2, keepdim=False)
        x = torch.cat([visual_feature.view(batch_size, -1), 
                       language_feature,
                       proprio], dim=-1)
        x = self.in_proj(x)
        for block, ln in zip(self.blocks, self.ln): x = x + block(ln(x))
        return self.out_proj(x).view(batch_size, self.num_action_chunk, -1)


class BaseModel(nn.Module):
    def __init__(self,
                 vision_backbone = "resnet18.a1_in1k",
                 model_type = "continuous",
                 dim_language = 768,
                 dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                 dim_actions = 20, # 14 for euler angles
                 num_action_chunk = 10,
                 action_scale = 100,
                 num_bins = 256,
                 **kwargs
                 ):
        super().__init__()
        self.action_scale = action_scale
        self.model_type = model_type
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        self.num_bins = num_bins
        print('Number of num_bins', num_bins)
        assert model_type in ['continuous', 'discrete', 'flow-matching']
        self.vision_backbone = create_model(vision_backbone, pretrained=True)
        del self.vision_backbone.fc
        self.decoder = MlpDecoder(
                                    depth = 3,
                                    hidden_size = 512,
                                    mlp_ratio = 4.0,
                                    dim_visual = self.vision_backbone.num_features,
                                    dim_language = dim_language,
                                    num_views = 1,
                                    dim_actions = dim_actions,
                                    dim_proprio = dim_proprio,
                                    num_action_chunk = num_action_chunk,
                                    num_bins = num_bins)
        if model_type == 'discrete': self.loss = nn.CrossEntropyLoss()
        else: self.loss = nn.HuberLoss(delta=0.1)

    def forward(self,
                images: torch.FloatTensor, # B * V * C * H * W,
                encoded_language: torch.Tensor, # B C
                proprio: torch.Tensor, # B C
                action_seq: torch.Tensor):
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        output_action = self.decoder(
            visual_feature = vision_embedding,
            language_feature = encoded_language,
            proprio = proprio
        )
        if self.model_type == 'discrete':
            return self.loss(output_action.view(-1, self.num_bins), action_seq.view(-1))
        else:
            return self.loss(output_action, action_seq * self.action_scale)
        
    def pred_action(self,
                images: torch.Tensor, # B V C H W
                encoded_language: torch.Tensor, # B C
                proprio: torch.Tensor
            ):
        print('xxxxxxxxxxxxxxxxxx image', images.shape, encoded_language.shape, proprio.shape)
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        if self.model_type == 'continuous': 
            pred_action = self.decoder(      
                        visual_feature = vision_embedding,
                        language_feature = encoded_language,
                        proprio = proprio)
            pred_action /= self.action_scale
        elif self.model_type == 'discrete':
            pred_action = self.decoder(      
                    visual_feature = vision_embedding,
                    language_feature = encoded_language,
                    proprio = proprio).view(B, self.num_action_chunk, self.dim_actions, self.num_bins).argmax(dim=-1)
        return pred_action

## Continuous Models
@register_model
def model_abs_ee_cnt(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_abs_qpos_cnt(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_rel_ee_cnt(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 1,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_cnt(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 1,
        num_bins = 1
    )
    return model, language_encoder()

## Discrete Models
@register_model
def model_abs_ee_dis(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "discrete",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 256
    )
    return model, language_encoder()

@register_model
def model_abs_qpos_dis(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "discrete",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 256
    )
    return model, language_encoder()

@register_model
def model_rel_ee_dis(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "discrete",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 256
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_dis(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "discrete",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 256
    )
    return model, language_encoder()


# if __name__ == "__main__":
#     model, lang_encoder = model_base(num_action_chunk = 10)
    
#     images = torch.randn(1, 1, 3, 224, 224) # B * V * C * H * W,
#     text = "test"
#     encoded_language = lang_encoder([text]) # B C
#     proprio = torch.randn(1, 20) # B C
#     action_seq = torch.randn(1, 10, 20)
#     print("start infer")

#     output = model(
#         images = images,
#         proprio = proprio,
#         action_seq = action_seq,
#         **encoded_language
#     )
    
#     print(output)