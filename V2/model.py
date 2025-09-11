import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models import create_model
import timm.models
from typing import List
from transformers import AutoTokenizer, SiglipTextModel
from timm.models.vision_transformer import Mlp
import math

print("model init")


def basic_init(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)    

class Normalizer(nn.Module):
    def __init__(self, dim_proprio=14, dim_actions=14, normalize_proprio=False, normalize_action=True):
        super().__init__()
        self.normalize_proprio = normalize_proprio
        self.normalize_action = normalize_action

        # correct shapes
        self.register_buffer("proprio_mean", torch.zeros(dim_proprio))
        self.register_buffer("proprio_std", torch.ones(dim_proprio))
        self.register_buffer("action_mean", torch.zeros(dim_actions))
        self.register_buffer("action_std", torch.ones(dim_actions))

    def set_dataset_stats(self, mean, std):
        self.proprio_mean.copy_(torch.tensor(mean["proprio"]))
        self.proprio_std.copy_(torch.tensor(std["proprio"]))
        self.action_mean.copy_(torch.tensor(mean["action"]))
        self.action_std.copy_(torch.tensor(std["action"]))

    def normalize(self, proprio, action_seq=None):
        if self.normalize_proprio and proprio is not None:
            proprio = (proprio - self.proprio_mean[None]) / (self.proprio_std[None] + 1e-6)

        if self.normalize_action and action_seq is not None:
            action_seq = (action_seq - self.action_mean[None, None, :]) / (self.action_std[None, None, :] + 1e-6)

        return proprio, action_seq

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
    def __init__(self, pt_path = "lang_emd_cube_cup.pt"):
        self.language_emb = torch.load(pt_path, map_location="cpu")
        print(f"successfully load language hub from {pt_path}: {self.language_emb.keys()}")
        
    @torch.no_grad()
    def encode_language(self, language_inputs: str):
        return self.language_emb[language_inputs]

class MlpDecoder(nn.Module):
    def __init__(self, 
                 model_type,
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
        self.model_type = model_type
        if model_type == 'flow-matching': 
            self.time_encoder = TimeEmbedder(hidden_size // 4)
            in_dim = dim_visual * num_views + dim_language + dim_proprio \
                    + dim_actions * num_action_chunk + hidden_size // 4
        else:
            in_dim = dim_visual * num_views + dim_language + dim_proprio
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

        if model_type == 'discrete':
            self.out_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, dim_actions * num_action_chunk * num_bins)
            )
        else:
            self.out_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, dim_actions * num_action_chunk)
            )
        self.apply(basic_init)
    
    def forward(self, 
                visual_feature, # B num_views num_tokens dim_visual
                language_feature, # B dim_language
                proprio, # B dim_proprio
                noise_action = None, # B num_action_chunk dim_action
                t = None # B
        ):
        batch_size = visual_feature.shape[0]
        visual_feature = torch.mean(visual_feature, dim=-2, keepdim=False)
        x = torch.cat([visual_feature.view(batch_size, -1), 
                                language_feature,
                                proprio], dim=-1)
        if self.model_type == 'flow-matching':
            x = torch.cat([x, noise_action.view(batch_size, -1), self.time_encoder(t)], dim =-1)            
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
                    depth = 3,
                    num_views = 1,
                    normalize_proprio = False,
                    normalize_action = False,
                 **kwargs
                 ):
        super().__init__()
        self.normalizer = Normalizer(dim_proprio=dim_proprio, dim_actions=dim_actions, normalize_proprio=normalize_proprio, normalize_action=normalize_action)

        self.action_scale = action_scale
        self.model_type = model_type
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        print('num_bins:', num_bins)
        if model_type == 'discrete':
            assert num_bins > 0, "num_bins must be greater than 0 for discrete models"
        else:
            assert num_bins == 1, "num_bins must be 1 for continuous models"
        self.num_bins = num_bins
        print('Number of num_bins', num_bins)
        assert model_type in ['continuous', 'discrete', 'flow-matching']
        self.vision_backbone = create_model(vision_backbone, pretrained=False)
        del self.vision_backbone.fc
        self.decoder = MlpDecoder(
                                    model_type = model_type,
                                    depth = depth,
                                    hidden_size = 512,
                                    mlp_ratio = 4.0,
                                    dim_visual = self.vision_backbone.num_features,
                                    dim_language = dim_language,
                                    num_views = num_views ,
                                    dim_actions = dim_actions,
                                    dim_proprio = dim_proprio,
                                    num_action_chunk = num_action_chunk,
                                    num_bins = self.num_bins,
                                    )
        if model_type == 'discrete': self.loss = nn.CrossEntropyLoss()
        else: self.loss = nn.HuberLoss(delta=0.1)

    def forward(self,
                images: torch.FloatTensor, # B * V * C * H * W,
                encoded_language: torch.Tensor, # B C
                proprio: torch.Tensor, # B C
                action_seq: torch.Tensor):
        proprio, action_seq = self.normalizer.normalize(proprio, action_seq)

        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        # Flow Matching
        noise_action = None
        t = None
        if self.model_type == 'flow-matching':
            t = (torch.rand(1, device=images.device) + torch.arange(images.shape[0], device=images.device) / images.shape[0]) % (1 - 1e-5)
            noise = torch.randn_like(action_seq)
            noise_action = noise * t.view(-1, 1, 1) + action_seq * (1 - t).view(-1, 1, 1)
            output_action = self.decoder(
                visual_feature = vision_embedding,
                language_feature = encoded_language,
                proprio = proprio,
                noise_action = noise_action, # B num_action_chunk dim_action
                t = t # B
            )
        else:
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
                proprio: torch.Tensor,
                steps = 5
            ):
        # print('xxxxxxxxxxxxxxxxxx image', images.shape, encoded_language.shape, proprio.shape)
        proprio, action_seq = self.normalizer.normalize(proprio, action_seq)
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
        elif self.model_type == 'flow-matching':
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
                action_with_noise = action_with_noise - (action_with_noise - pred_action / self.action_scale) / time.view(B, 1, 1) / steps
            return action_with_noise # denoised action
        return pred_action

def get_positional_embeddings(seq_length, d_model):
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_length, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe

class ACT_Decoder(nn.Module):
    def __init__(self,
                hidden_dim = 512,
                dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 30,

                vision_backbone = "resnet18.a1_in1k",
                dim_language = 768,

                action_scale = 100,
                num_views = 1,
                normalize_proprio = False,
                normalize_action = False
                ):

        super().__init__()
        self.action_scale = action_scale
        self.normalizer = Normalizer(dim_proprio=dim_proprio, dim_actions=dim_actions, normalize_proprio=normalize_proprio, normalize_action=normalize_action)
        self.num_action_chunk = num_action_chunk
        self.proprio_proj = nn.Linear(dim_proprio, hidden_dim)
        self.lang_proj = nn.Linear(dim_language, hidden_dim)
        
        self.vision_backbone = create_model(vision_backbone, pretrained=False)
        del self.vision_backbone.fc

        self.action_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, dim_actions))
        
        self.queries = nn.Parameter(torch.zeros(1, num_action_chunk, hidden_dim), requires_grad=True)
        self.queries_pos_emb = nn.Parameter(get_positional_embeddings(num_action_chunk, hidden_dim), requires_grad=False)
        self.input_pos_emb = nn.Parameter(get_positional_embeddings(49*num_views+2, hidden_dim), requires_grad=False)

        assert hidden_dim % 64 == 0
        self.model = nn.Transformer(
            d_model=hidden_dim,
            nhead=hidden_dim//64,
            num_encoder_layers = 4,
            num_decoder_layers = 2,
            dim_feedforward = hidden_dim * 4,
            dropout = 0.0,
            batch_first = True,
            norm_first = False,
        )
        self.loss = nn.HuberLoss(delta=0.1)

    def forward(self, 
                images: torch.Tensor,  # B V N C
                encoded_language: torch.Tensor, # B, ua_dim
                proprio: torch.Tensor, # B, prio_dim
                action_seq: torch.Tensor):
        proprio, action_seq = self.normalizer.normalize(proprio, action_seq)
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)

        inputs = torch.cat(
            [
                vision_embedding.flatten(start_dim=1, end_dim=2),
                self.lang_proj(encoded_language).unsqueeze(1),
                self.proprio_proj(proprio).unsqueeze(1)
            ], dim = 1
        )
        inputs = inputs + self.input_pos_emb
        query = self.queries.repeat(B, 1, 1) + self.queries_pos_emb
        
        output = self.model.forward(inputs, query) # B ac hidden
        output = self.action_head(output) # B ac 14
        return self.loss(output, action_seq * self.action_scale)

    def pred_action(self,
                images: torch.Tensor, # B V C H W
                encoded_language: torch.Tensor, # B C
                proprio: torch.Tensor,
                steps = 5
            ):
        proprio, action_seq = self.normalizer.normalize(proprio, action_seq)
        B, V, C, H, W = images.shape
        vision_embedding = self.vision_backbone.forward_features(images.view(B*V, C, H, W)) # B num_features H W
        vision_embedding = vision_embedding.flatten(start_dim=-2) # B*V num_features N
        _, num_features, N = vision_embedding.shape
        vision_embedding = vision_embedding.permute(0, 2, 1).view(B, V, N, num_features)
        inputs = torch.cat(
                    [
                        vision_embedding.flatten(start_dim=1, end_dim=2),
                        self.lang_proj(encoded_language).unsqueeze(1),
                        self.proprio_proj(proprio).unsqueeze(1)
                    ], dim = 1
                )
        inputs = inputs + self.input_pos_emb
        query = self.queries.repeat(B, 1, 1) + self.queries_pos_emb
        
        output = self.model.forward(inputs, query) # B ac hidden
        output = self.action_head(output) 
        return output
        
@register_model
def ACT_3RGB_14DoFs_14Proprio_chunk30(pt_path = "encoded_language.pt",
                                        **kwargs):
    return ACT_Decoder(dim_proprio = 14,
                dim_actions = 14), language_encoder(pt_path=pt_path)

@register_model
def ACT_3RGB_20DoFs_20Proprio_chunk30(pt_path = "encoded_language.pt",
                                        **kwargs):
    return ACT_Decoder(dim_proprio = 20,
                dim_actions = 20), language_encoder(pt_path=pt_path)

## Continuous Models
@register_model
def model_abs_ee_cnt(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                pt_path = "encoded_language.pt",
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
    return model, language_encoder(pt_path=pt_path)

@register_model
def model_abs_ee_cnt_act30(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 30,
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
def model_abs_ee_cnt_act60(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 60,
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

## Continuous Models
@register_model
def model_abs_ee_cnt_mlp1(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                pt_path = "encoded_language.pt",
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 1,
        depth = 1
    )
    return model, language_encoder(pt_path=pt_path)

@register_model
def model_abs_ee_cnt_mlp6(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 6
    )
    return model, language_encoder()

@register_model
def model_abs_ee_cnt_mlp9(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 9
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
def model_abs_qpos_cnt_act30(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 30,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 1,
        num_views=3
    )
    return model, language_encoder()

@register_model
def model_abs_qpos_cnt_act60(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 60,
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
def model_abs_qpos_cnt_mlp1(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 1
    )
    return model, language_encoder()

@register_model
def model_abs_qpos_cnt_mlp6(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 6
    )
    return model, language_encoder()

@register_model
def model_abs_qpos_cnt_mlp9(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 9
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
def model_rel_ee_cnt_act30(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 30,
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
def model_rel_ee_cnt_act60(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 60,
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
def model_rel_ee_cnt_mlp1(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 1
    )
    return model, language_encoder()

@register_model
def model_rel_ee_cnt_mlp6(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 6
    )
    return model, language_encoder()

@register_model
def model_rel_ee_cnt_mlp9(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 9
    )
    return model, language_encoder()


@register_model
def model_rel_ee_cnt_act30(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 30,
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
def model_rel_ee_cnt_act60(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 60,
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
def model_rel_ee_cnt_mlp1(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 1
    )
    return model, language_encoder()

@register_model
def model_rel_ee_cnt_mlp6(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 6
    )
    return model, language_encoder()

@register_model
def model_rel_ee_cnt_mlp9(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 9
    )
    return model, language_encoder()


@register_model
def model_rel_qpos_cnt(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                pt_path = "encoded_language.pt",
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
    return model, language_encoder(pt_path=pt_path)

@register_model
def model_rel_qpos_cnt_act30(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 30,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "continuous",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 1,
        num_bins = 1,
        num_views = 3
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_cnt_act60(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 60,
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
def model_rel_qpos_cnt_mlp1(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 1
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_cnt_mlp6(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 6
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_cnt_mlp9(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
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
        num_bins = 1,
        depth = 9
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
        action_scale = 1, # not used in discrete model
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
        action_scale = 1,
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
        action_scale = 1,
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
        action_scale = 1,
        num_bins = 256
    )
    return model, language_encoder()

@register_model
def model_abs_ee_flow(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "flow-matching",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_abs_qpos_flow(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "flow-matching",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 100,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_rel_ee_flow(dim_proprio = 20, # 14 for euler angles, 20 for rot6d
                dim_actions = 20, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "flow-matching",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 1,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_flow(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 10,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "flow-matching",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 1,
        num_bins = 1
    )
    return model, language_encoder()

@register_model
def model_rel_qpos_flow_act30(dim_proprio = 14, # 14 for euler angles, 20 for rot6d
                dim_actions = 14, # 14 for euler angles
                num_action_chunk = 30,
                **kwargs):
    model = BaseModel(
        vision_backbone = "resnet18.a1_in1k",
        model_type = "flow-matching",
        dim_language = 768,
        dim_proprio = dim_proprio, # 14 for euler angles, 20 for rot6d
        dim_actions = dim_actions, # 14 for euler angles
        num_action_chunk = num_action_chunk,
        action_scale = 1,
        num_bins = 1
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