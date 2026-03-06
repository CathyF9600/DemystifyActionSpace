import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
import numpy as np
import timm
from timm.models.registry import register_model

norm_layer = {"bn": nn.BatchNorm2d, 
              "gn": nn.GroupNorm}

pooling_layer = {"avg": nn.AdaptiveAvgPool2d, 
                 "max": nn.AdaptiveMaxPool2d}


class ResNet(nn.Module):
    def __init__(self, 
        model_name: str = "resnet18",
        pretraine_path: str = None,
        pretrained: bool = True,
        input_dim: int = 3,
        norm_type: str = "bn",
        pooling_type: str = "avg",
        add_spatial_coordinates: bool = False,
        use_alpha_channel: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)


        self.model = timm.create_model(model_name, 
                                    pretrained=True)


        self.norm_type = norm_type
        assert norm_type in norm_layer
        self.norm_layer = norm_layer[norm_type]

        assert pooling_type in pooling_layer
        self.pooling_layer = pooling_layer[pooling_type]

        if norm_type != "bn":
                self._replace_bn()

        del self.model.fc

        # add spatial information to the image
        self.add_spatial_coordinates = add_spatial_coordinates
        self.use_alpha_channel = use_alpha_channel
        self.c_num = input_dim

        if self.add_spatial_coordinates:
            self.spatial_coordinates = AddSpatialCoordinates(dtype=self.model.conv1.weight.dtype)
            self.c_num += 2
        if self.use_alpha_channel:
            self.c_num += 1
        if self.c_num != 3:
            self.model.conv1 = nn.Conv2d(self.c_num, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.num_channels = 512 if model_name in ('resnet18', 'resnet34') else 2048

    def forward(self, img):
        if self.add_spatial_coordinates:
            img = self.spatial_coordinates(img)
        output = self.model(img)
        return output

    def forward_stem(self, img):
        if self.add_spatial_coordinates:
            img = self.spatial_coordinates(img)
        output = self.model.conv1(img)
        output = self.model.bn1(output)
        output = self.model.act1(output)
        output = self.model.maxpool(output)
        return output

    # def forward_head(self, image_feature):
    #     image_feature = self.model.global_pool(image_feature)
    #     image_feature = self.model.fc(image_feature)
    #     return image_feature

    def get_visual_blocks(self):
        visual_blocks = []
        for name, child in self.model.named_children():
            if isinstance(child, nn.Sequential):
                for visual_block in child:
                    visual_blocks.append(visual_block)
        return visual_blocks

    def feature_info(self):
        idx = 0
        feature_info = self.model.feature_info[1:]
        for name, child in self.model.named_children():
            if isinstance(child, nn.Sequential):
                feature_info[idx]['num_blocks'] = len(child)
                idx += 1
        return feature_info

    def _replace_bn(self):
        root_module = self.model
        bn_list = [k.split('.') for k, m in self.model.named_modules() if isinstance(m, nn.BatchNorm2d)]

        for *parent, k in bn_list:
            parent_module = root_module
            if len(parent) > 0:
                parent_module = root_module.get_submodule('.'.join(parent))
            if isinstance(parent_module, nn.Sequential):
                src_module = parent_module[int(k)]
            else:
                src_module = getattr(parent_module, k)

            tgt_module = self.norm_layer(src_module.num_features//16, src_module.num_features)

            if isinstance(parent_module, nn.Sequential):
                parent_module[int(k)] = tgt_module
            else:
                setattr(parent_module, k, tgt_module)

        # verify that all BN are replaced
        bn_list = [k.split('.') for k, m in self.model.named_modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bn_list) == 0


class AddSpatialCoordinates(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(AddSpatialCoordinates, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        grids = [
            torch.linspace(-1, 1, steps=s, device=x.device, dtype=self.dtype) 
            for s in x.shape[-2:]  # add spatial coordinates with HxW shape
        ]

        grid = torch.meshgrid(grids, indexing='ij')
        grid = torch.stack(grid, dim=0)

        # reshape to B*F*V, 2, H, W
        BFV, *_ = x.shape
        grid = grid.expand(BFV, *grid.shape)

        # cat on the channels dimension
        return torch.cat([x, grid], dim=-3)


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class FiLM_layer(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        dim: int,
    ):
        super().__init__()
        self.net = Mlp(condition_dim, dim * 2, dim * 2)
        self.apply(init_weight)
        nn.init.zeros_(self.net.fc2.weight)
        nn.init.zeros_(self.net.fc2.bias)

    def forward(self, conditions, hiddens):
        # conditions shape: B, C'
        # hidden shape: B, C, H, W or B, N, C
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        assert hiddens.shape[0] == scale.shape[0]
        if len(hiddens.shape) == 4:
            assert scale.shape[-1] == hiddens.shape[1]
            scale = scale.unsqueeze(-1).unsqueeze(-1) # shape -> B, C, 1, 1
            shift = shift.unsqueeze(-1).unsqueeze(-1) # shape -> B, C, 1, 1
        elif len(hiddens.shape) == 3:
            assert scale.shape[-1] == hiddens.shape[-1]
            scale = scale.unsqueeze(1) # shape -> B, 1, C
            shift = shift.unsqueeze(1)
        return hiddens * (scale + 1) + shift

class FilmResNet(nn.Module):
    def __init__(
        self,
        image_dim: int=3,
        cond_dim: int=1024,
        backbone_name: str = "resnet34"
    ):
        super().__init__()
        self.backbone = ResNet(model_name= backbone_name, pretrained= True, input_dim = image_dim)
        self.vision_dim = self.backbone.num_channels

        feature_info = self.backbone.feature_info()
        self.film_layer = nn.ModuleList()
        for info in feature_info:
            feature_dim = info['num_chs']
            feature_depth = info['num_blocks']
            self.film_layer.extend([FiLM_layer(cond_dim, feature_dim) for _ in range(feature_depth)])

    def forward(self, image, condition):
        # image B C H W
        # condition B D
        image_feature = self.backbone.forward_stem(image)
        visual_blocks = self.backbone.get_visual_blocks()
        for visual_block, film_layer in zip(visual_blocks, self.film_layer):
            image_feature = visual_block(image_feature)
            image_feature = film_layer(condition, image_feature)

        # image_feature = self.backbone.forward_head(image_feature)
        return image_feature
    
## Continuous Models
@register_model
def FiLM_Resnet18(
                cond_dim: int=768,
                backbone_name: str = "resnet18",
                **kwargs):
    return FilmResNet(
        cond_dim = cond_dim,
        backbone_name = backbone_name
    )
