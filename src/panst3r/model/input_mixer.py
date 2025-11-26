# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import torch.nn as nn

from croco.models.blocks import Block
from must3r.model.blocks.pos_embed import get_pos_embed

class InputMixer(nn.Module):
    def __init__(self, img_size, patch_size, in_dim, hidden_dim, num_heads=12, num_layers=3, ff_dim_mult=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.max_seq_len = max(img_size)//patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.rope = get_pos_embed('RoPE100')

        self.mixer_blk = nn.ModuleList([
            Block(hidden_dim, num_heads, mlp_ratio=ff_dim_mult, rope=self.rope, qkv_bias=True)
            for i in range(num_layers)])
        self.mixer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, pos):
        x = self.in_proj(x)
        for blk in self.mixer_blk:
            x = blk(x, pos)
        out = self.mixer_norm(x)

        return out
