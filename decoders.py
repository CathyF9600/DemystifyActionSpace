import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, Mlp
import math


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
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)        
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        
    def forward(self, x, c, t):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))  
        x = x + self.cross_attn(self.norm2(x), c, c)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x

class TransfomerDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)        
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.1)
        
    def forward(self, x, c):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), c, c)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransfomerEncoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)    
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=0.1)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 model_type,
                 encoder_depth = 3,
                 decoder_depth = 3,
                 hidden_size = 512,
                 num_heads = 8,
                 dim_visual = 512,
                 dim_language = 768,
                 dim_proprio = 14,
                 dim_actions = 14,
                 num_action_chunk = 5,
                 num_bins = 256
                 ):
        super().__init__()
        self.model_type = model_type
        self.dim_visual = dim_visual
        self.encoder = nn.ModuleList([TransfomerEncoderBlock(hidden_size = hidden_size, 
                                                             num_heads=num_heads) for _ in range(encoder_depth)])
        
        self.queries = nn.Parameter(torch.randn(1, num_action_chunk, hidden_size))
        self.visual_proj = nn.Linear(dim_visual, hidden_size)
        self.language_proj = nn.Linear(dim_language, hidden_size)
        self.proprio_proj = nn.Linear(dim_proprio, hidden_size)
        if model_type == 'discrete':
            self.out_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, dim_actions * num_bins)
            )
        else:
            self.out_proj = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, dim_actions)
            )
        if model_type == 'flow-matching':
            self.action_in_proj = nn.Linear(dim_actions, hidden_size)
            self.time_encoder = TimeEmbedder(hidden_size)
            self.decoder = nn.ModuleList([DiTBlock(hidden_size = hidden_size, 
                                            num_heads=num_heads) for _ in range(decoder_depth)])
        else:
            self.decoder = nn.ModuleList([TransfomerDecoderBlock(hidden_size = hidden_size, 
                                            num_heads=num_heads) for _ in range(decoder_depth)])
        self.apply(basic_init)
        
    def forward(self, 
                visual_feature, # B num_views num_tokens dim_visual
                language_feature, # B dim_language
                proprio, # B dim_proprio
                noise_action = None, # B num_action_chunk dim_action
                t = None # B
            ):
        batch_size = visual_feature.shape[0]
        visual_feature = self.visual_proj(visual_feature.view(batch_size, -1, self.dim_visual))
        language_feature = self.language_proj(language_feature).unsqueeze(1)
        proprio = self.proprio_proj(proprio).unsqueeze(1)
        c = torch.cat([visual_feature, language_feature, proprio], dim = 1)
        for block in self.encoder: c = block(c)
        x = self.queries.repeat(batch_size, 1, 1)
        t = self.time_encoder(t)
        if self.model_type == 'flow-matching': 
            x = x + self.action_in_proj(noise_action)
            for block in self.decoder: x = block(x, c, t)
        else:
            for block in self.decoder: x = block(x, c)
        return self.out_proj(x)
        
class MlpDecoder(nn.Module):
    def __init__(self, 
                 model_type,
                 depth = 2,
                 hidden_size = 512,
                 mlp_ratio = 4.0,
                 dim_visual = 512,
                 dim_language = 768,
                 num_views = 3,
                 dim_actions = 20,
                 dim_proprio = 20,
                 num_action_chunk = 5,
                 num_bins = 256
                 ):
        super().__init__()
        self.model_type = model_type
        self.num_action_chunk = num_action_chunk
        self.dim_actions = dim_actions
        if model_type == 'flow-matching': 
            self.time_encoder = TimeEmbedder(hidden_size // 4)
            self.in_proj = nn.Linear(dim_visual * num_views + dim_language + dim_proprio + dim_actions * num_action_chunk + hidden_size // 4,
                                    hidden_size)
        else:
            self.in_proj = nn.Linear(dim_visual * num_views + dim_language + dim_proprio,
                                    hidden_size)
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
        for block, ln in zip(self.blocks, self.ln):x = x + block(ln(x))
        return self.out_proj(x).view(batch_size, self.num_action_chunk, -1)
    

@register_model
def mlp_decoder_base(model_type, 
                     dim_visual,
                     dim_language,
                     dim_proprio,
                     dim_actions, 
                     num_action_chunk,
                     num_bins,
                     **kwarges):
    return MlpDecoder(
                model_type = model_type,
                depth = 2,
                hidden_size = 512,
                mlp_ratio = 4.0,
                dim_visual = dim_visual,
                dim_language = dim_language,
                num_views = 3,
                dim_proprio = dim_proprio,
                dim_actions = dim_actions,
                num_action_chunk = num_action_chunk,
                num_bins = num_bins
            )

@register_model
def transformer_decoder_base(model_type, 
                     dim_visual,
                     dim_language,
                     dim_proprio,
                     dim_actions, 
                     num_action_chunk, 
                     num_bins,
                     **kwarges):
    return TransformerDecoder(
                model_type = model_type,
                encoder_depth = 3,
                decoder_depth= 3,
                hidden_size = 512,
                num_heads= 8,
                dim_visual = dim_visual,
                dim_language = dim_language,
                dim_proprio = dim_proprio,
                dim_actions = dim_actions,
                num_action_chunk = num_action_chunk,
                num_bins = num_bins
            )

