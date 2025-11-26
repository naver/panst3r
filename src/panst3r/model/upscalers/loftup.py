"""Upscaler implementation based on the LoftUp upscaler: https://github.com/andrehuang/loftup (MIT License)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from panst3r.model.blocks import CrossonlyDecoderBlock

class MinMaxScaler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = x.shape[1]
        flat_x = x.permute(1, 0, 2, 3).reshape(c, -1)
        flat_x_min = flat_x.min(dim=-1).values.reshape(1, c, 1, 1)
        flat_x_scale = flat_x.max(dim=-1).values.reshape(1, c, 1, 1) - flat_x_min
        return ((x - flat_x_min) / flat_x_scale.clamp_min(0.0001)) - .5

class ImplicitFeaturizer(torch.nn.Module):

    def __init__(self, color_feats=True, n_freqs=10, learn_bias=False, time_feats=False, lr_feats=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color_feats = color_feats
        self.time_feats = time_feats
        self.n_freqs = n_freqs
        self.learn_bias = learn_bias

        self.dim_multiplier = 2

        if self.color_feats:
            self.dim_multiplier += 3

        if self.time_feats:
            self.dim_multiplier += 1

        if self.learn_bias:
            self.biases = torch.nn.Parameter(torch.randn(2, self.dim_multiplier, n_freqs).to(torch.float32))

        self.low_res_feat = lr_feats

    def forward(self, original_image):
        b, c, h, w = original_image.shape
        grid_h = torch.linspace(-1, 1, h, device=original_image.device)
        grid_w = torch.linspace(-1, 1, w, device=original_image.device)
        feats = torch.cat([t.unsqueeze(0) for t in torch.meshgrid([grid_h, grid_w])]).unsqueeze(0)
        feats = torch.broadcast_to(feats, (b, feats.shape[1], h, w))

        if self.color_feats:
            feat_list = [feats, original_image]
        else:
            feat_list = [feats]

        feats = torch.cat(feat_list, dim=1).unsqueeze(1)
        freqs = torch.exp(torch.linspace(-2, 10, self.n_freqs, device=original_image.device)) \
            .reshape(1, self.n_freqs, 1, 1, 1) # torch.Size([1, 30, 1, 1, 1])
        feats = (feats * freqs) # torch.Size([1, 30, 5, 224, 224])

        if self.learn_bias:
            sin_feats = feats + self.biases[0].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
            cos_feats = feats + self.biases[1].reshape(1, self.n_freqs, self.dim_multiplier, 1, 1) # torch.Size([1, 30, 5, 224, 224])
        else:
            sin_feats = feats
            cos_feats = feats

        sin_feats = sin_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])
        cos_feats = cos_feats.reshape(b, self.n_freqs * self.dim_multiplier, h, w) # torch.Size([1, 150, 224, 224])

        if self.color_feats:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats), original_image]
        else:
            all_feats = [torch.sin(sin_feats), torch.cos(cos_feats)]

        if self.low_res_feat is not None:
            upsampled_feats = F.interpolate(self.low_res_feat, size=(h, w), mode='bilinear', align_corners=False)
            all_feats.append(upsampled_feats)

        return torch.cat(all_feats, dim=1)

class LoftUpUpscaler(nn.Module):
    """Builds a FPN for panoptic segmentation."""

    def __init__(self, input_dim, dim, output_stride=2, patch_size=16, color_feats=True, n_freqs=20, num_heads=4, num_layers=2, lr_pe_type="sine"):
        super().__init__()

        self.output_stride = output_stride
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(input_dim, input_dim, kernel_size=1)

        if color_feats:
            start_dim = 5 * n_freqs * 2 + 3
        else:
            start_dim = 2 * n_freqs * 2

        num_patches = patch_size * patch_size
        self.lr_pe_type = lr_pe_type
        if self.lr_pe_type == "sine":
            self.lr_pe = ImplicitFeaturizer(color_feats=False, n_freqs=5, learn_bias=True)
            self.lr_pe_dim = 2 * 5 * 2
            concat_dim = input_dim + self.lr_pe_dim

        elif self.lr_pe_type == "learnable":
            self.lr_pe = nn.Parameter(torch.randn(1, num_patches, dim))
            self.lr_pe_dim = dim
            concat_dim = input_dim

        else:
            raise ValueError(f"Unknown lr_pe_type: {self.lr_pe_type}")


        self.lr_input_proj = nn.Sequential(
            nn.Linear(concat_dim, dim),
            nn.LayerNorm(dim),
        )

        self.fourier_feat = torch.nn.Sequential(
                                MinMaxScaler(),
                                ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
                            )

        self.first_conv = torch.nn.Sequential(
                            nn.GroupNorm(num_groups=1, num_channels=start_dim),
                            nn.Conv2d(start_dim, dim, kernel_size=3, padding=1),
                            nn.GroupNorm(num_groups=8, num_channels=dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                            nn.GroupNorm(num_groups=8, num_channels=dim),
                            nn.ReLU(inplace=True),
                        )

        self.ca_transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.ca_transformer_blocks.append(
                CrossonlyDecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=1)
            )
        self.ca_transformer_norm = nn.LayerNorm(dim)

    def forward(self, inputs, img_shape):
        lr_feats, img = inputs
        H, W = img_shape

        lr_feats = lr_feats.transpose(-1,-2).view(lr_feats.shape[0], -1, H//self.patch_size, W//self.patch_size) # to 2d
        patch_feats = self.patch_embed(lr_feats)

        Cl, Hl, Wl = lr_feats.shape[1:]

        if H > W:
            # Need to transpose guidance images
            img = img.transpose(2, 3)

        if self.output_stride != 1:
            # Rescale guidance to match the output size
            img = F.interpolate(img, scale_factor=1./self.output_stride, mode='bilinear', align_corners=False)
        x = self.fourier_feat(img)
        x = self.first_conv(x)
        B, Ch, Hout, Wout = x.shape
        x = x.flatten(2).transpose(-1,-2)

        if self.lr_pe_type == "sine":
            lr_pe = self.lr_pe(lr_feats)
            lr_feats_pe = torch.cat([lr_feats, lr_pe], dim=1)
            lr_feats_pe = lr_feats_pe.flatten(2).permute(0, 2, 1)

        elif self.lr_pe_type == "learnable":
            lr_feats = lr_feats.flatten(2).permute(0, 2, 1) # (B, H*W, C)
            if lr_feats.shape[1] != self.lr_pe.shape[1]:
                len_pos_old = int(math.sqrt(self.lr_pe.shape[1]))
                pe = self.lr_pe.reshape(1, len_pos_old, len_pos_old, Cl).permute(0, 3, 1, 2)
                pe = F.interpolate(pe, size=(Hl, Wl), mode='bicubic', align_corners=False)
                pe = pe.reshape(1, Cl, Hl*Wl).permute(0, 2, 1)
                lr_feats_pe = lr_feats + pe
            else:
                lr_feats_pe = lr_feats + self.lr_pe

        else:
            raise ValueError(f"Unknown lr_pe_type: {self.lr_pe_type}")

        lr_feats_pe = self.lr_input_proj(lr_feats_pe)
        for blk in self.ca_transformer_blocks:
            x,_ = blk(x, lr_feats_pe, None, None)
        x = self.ca_transformer_norm(x)

        # Reshape back to (B, C, H, W)
        x = x.transpose(-1,-2).reshape(B, Ch, Hout, Wout)

        fpn = [patch_feats] # Used in cross-attention
        mask_feats = x

        return fpn, mask_feats
