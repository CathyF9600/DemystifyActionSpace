import torch
import torch.utils.checkpoint
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, SiglipTextModel

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


encoder = language_encoder().to('cuda')
print('encoder', encoder)

target_dict = {}
text = ["touch cube", "pick cup", "pick cup and place it into the bowl", "pick the cube with the left hand and put it into the bowl with the right hand"]
for t in text:
    target_dict[t] = encoder([t]).squeeze().cpu()   
    print('t', t, target_dict[t].shape)

torch.save(target_dict, 'lang_emd_cube_cup.pt')