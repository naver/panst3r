# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
import torchvision.transforms as T


def get_dinov2_model(model_path='facebook/dinov2-large'):
    model = AutoModel.from_pretrained(model_path).eval()
    return model

def dinov2_transpose(model, activate=True):
    def wrapper_no(x, true_shape):
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'

        res = model(pixel_values=x).last_hidden_state
        return res

    def wrapper_yes(x: torch.Tensor, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return model(pixel_values=x).last_hidden_state
        if is_portrait.all():
            return model(pixel_values=x.transpose(2,3)).last_hidden_state

        # batch is a mix of both portraint & landscape
        l_result = model(pixel_values=x[is_landscape]).last_hidden_state
        p_result = model(pixel_values=x[is_portrait].transpose(2,3)).last_hidden_state

        result = l_result.new_zeros(B, *l_result.shape[1:])
        result[is_landscape] = l_result
        result[is_portrait] = p_result

        return result

    return wrapper_yes if activate else wrapper_no

class DinoV2Encoder(nn.Module):
    def __init__(self, dino_model='facebook/dinov2-large', output_stride=16, landscape_only=True):
        super().__init__()

        self.dinov2 = get_dinov2_model(dino_model)
        self.dinov2_magic = dinov2_transpose(self.dinov2, activate=landscape_only)
        self.embed_dim = self.dinov2.config.hidden_size
        self.norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.output_stride = output_stride

    def forward(self, image, true_shape):
        # DINOv2 normalize
        image = image * 0.5 + 0.5 # to [0,1]
        image = self.norm(image)

        # Scale so number of patches is the same
        h,w = [x // self.output_stride * self.dinov2.config.patch_size for x in image.shape[-2:]]
        x = F.interpolate(image, size=(h,w), mode='bilinear', align_corners=False)

        output = self.dinov2_magic(x, true_shape)
        out = output[:,1:] # remove the cls token

        return out

