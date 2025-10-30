# Copyright (C) 2025-present Naver Corporation. All rights reserved.
"""Mask-transformer-style panoptic segmentation head. Built on top of a feature pyramid neck."""

import torch
from torch import nn

from panst3r.model.text_encoder import TextEncoder
from panst3r.model.mask_transformer import MaskTransformer

from panst3r.utils import transpose_to_landscape, batched_map

def dummy_input_mixer(x, pos):
    """Dummy input mixer that does nothing. Used when no input mixer is provided."""
    return x

class PanopticDecoder(nn.Module):
    """Panoptic segmentation head inspired by mask transformers.
    Decodes a single set of object queries by attending to features from both views."""

    def __init__(self, input_mixer=None, upscaler=None, fpn_dim=[768], hidden_dim=768, mask_dim=256, ff_dim=2048, num_queries=200, num_heads=8, dec_layers=6, text_encoder='siglip', fixed_vocab=True, label_mode='sigmoid', two_stage=False, landscape_only=True, deep_supervision=True):
        super().__init__()
        assert upscaler is not None, 'Upscaler module must be provided'

        self.input_mixer = input_mixer if input_mixer is not None else dummy_input_mixer
        self.upscaler = upscaler
        self._upscaler_wrapper = transpose_to_landscape(upscaler, activate=landscape_only, dims=(2,3))
        self.text_encoder = TextEncoder(text_encoder, out_dim=hidden_dim, fixed_vocab=fixed_vocab)
        self.label_mode = label_mode

        if self.label_mode == 'softmax':
            self.nocls_token = nn.Parameter(torch.randn(self.text_encoder.embed_dim))

        self.mask_transformer = MaskTransformer(fpn_dim, hidden_dim, ff_dim, mask_dim, num_queries, num_heads, dec_layers, lang_dim=self.text_encoder.embed_dim, num_feature_levels=len(fpn_dim), two_stage=two_stage, landscape_only=landscape_only)

        self.deep_supervision = deep_supervision

    # def _pad_multi_ar(self, in_feats, in_imgs, pos, true_shape):

    #     return in_feats, in_imgs, pos, true_shape, pad_mask

    def forward(self, in_feats, in_imgs, pos, true_shape, classes, max_bs=None, outdevice=None, memory_queries=None, multi_ar=False):

        # Concat encoder, decoder and dino features
        if multi_ar:
            cat_feats = [torch.cat(tensors, dim=-1) for tensors in zip(*in_feats)]
        else:
            cat_feats = torch.cat(in_feats, dim=-1)

        # Prepare panst3r features
        def _process_fn(cat_feats, in_imgs, pos, true_shape):
            cat_feats = self.input_mixer(cat_feats, pos)
            fpn, mask_f = self._upscaler_wrapper((cat_feats, in_imgs), true_shape)

            return mask_f, *fpn

        mask_f, *fpn = batched_map(
            _process_fn,
            (cat_feats, in_imgs, pos, true_shape),
            flatten_dims=(0, 1),
            batch_size=max_bs,
            multi_ar=multi_ar,
        )

        cls_embeddings = self.text_encoder(classes)
        if self.label_mode == 'softmax':
            cls_embeddings = torch.cat([cls_embeddings, self.nocls_token[None]], dim=0)

        if memory_queries is None:
            pan_out = self.mask_transformer(fpn, mask_f, true_shape, cls_embeddings, max_bs=max_bs, deep_supervision=self.deep_supervision, outdevice=outdevice, multi_ar=multi_ar)
        else:
            outputs_class, outputs_masks, _ = self.mask_transformer.forward_prediction_heads(memory_queries, mask_f, cls_embeddings, max_bs=max_bs, outdevice=outdevice, multi_ar=multi_ar)

            pan_out = {
                'pred_logits': outputs_class,
                'pred_masks': outputs_masks,
            }
        return pan_out

