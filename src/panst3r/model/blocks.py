# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from croco.models.blocks import DropPath, Mlp, CrossAttention

class CrossonlyDecoderBlock(nn.Module):

    def __init__(self, dim, num_heads, pos_embed=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True,
                 masked_attention=False, is_pos_embed_for_sa_only=False, qkln=False):
        super().__init__()
        assert not qkln, "not implemented"
        if is_pos_embed_for_sa_only:
            assert pos_embed is None
        self.cross_attn = CrossAttention(dim, rope=pos_embed, num_heads=num_heads,
                                         qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.batch_drop_path_prob = -drop_path if drop_path < 0. else 0.
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos, mask_for_attention=None):
        y_ = self.norm_y(y)
        if self.batch_drop_path_prob == 0.0 or not self.training or torch.rand(1).item() >= self.batch_drop_path_prob:
            x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        if self.batch_drop_path_prob == 0.0 or not self.training or torch.rand(1).item() >= self.batch_drop_path_prob:
            x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y
