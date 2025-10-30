# Copyright (C) 2025-present Naver Corporation. All rights reserved.
"""Upscaler using a gradual 2x upscaling with a series of MLP + pixel shuffle layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from croco.models.blocks import Mlp

class PixelShuffleUpscaler(nn.Module):
    """Upscaler using a gradual 2x upscaling with a series of MLP + pixel shuffle layers"""

    def __init__(self, input_dim, patch_size=16, hidden_dim_factor=4, fp_dim=[768,512,384,256], fp_activation=nn.GELU, **kwargs):
        super().__init__()
        self.fp_dim = fp_dim
        self.patch_size = patch_size

        self.proj_8 = Mlp(in_features=input_dim, hidden_features=int(hidden_dim_factor*input_dim),
                          act_layer=fp_activation, out_features=fp_dim[1]*4)

        self.proj_4 = Mlp(in_features=fp_dim[1], hidden_features=int(hidden_dim_factor*fp_dim[1]),
                          act_layer=fp_activation, out_features=fp_dim[2]*4)

        self.proj_2 = Mlp(in_features=fp_dim[2], hidden_features=int(hidden_dim_factor*fp_dim[2]),
                          act_layer=fp_activation, out_features=fp_dim[3]*4)

        self.proj_16 = Mlp(in_features=input_dim, hidden_features=int(hidden_dim_factor*input_dim),
                           act_layer=fp_activation, out_features=fp_dim[0])

        self.fpn_strides = [16, 8, 4, 2]


    def forward(self, feats, img_shape):
        feats = feats[0] # Needed for transpose magic
        H, W = img_shape
        hs, ws = H // self.patch_size, W // self.patch_size # Size of patch features

        B = feats.shape[0]

        # FPN features
        f8 = self.proj_8(feats)
        f8 = f8.transpose(-1,-2).view(B, -1, hs, ws)
        f8 = F.pixel_shuffle(f8, 2)

        f4 = self.proj_4(f8.flatten(-2).transpose(-1,-2))
        f4 = f4.transpose(-1,-2).view(B, -1, hs*2, ws*2)
        f4 = F.pixel_shuffle(f4, 2)

        f2 = self.proj_2(f4.flatten(-2).transpose(-1,-2))
        f2 = f2.transpose(-1,-2).view(B, -1, hs*4, ws*4)
        f2 = F.pixel_shuffle(f2, 2)

        # Patch features
        f16 = self.proj_16(feats)
        f16 = f16.transpose(-1,-2).view(B, -1, hs, ws)

        fpn = [f16] # Used in cross-attention
        mask_feats = f2 # Used in mask prediction (dot product)

        return fpn, mask_feats
