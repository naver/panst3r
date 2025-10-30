# Adapted for PanSt3R from Mask2Former (https://github.com/facebookresearch/Mask2Former) by Meta Platforms, Inc.
# Original code licensed under the MIT License.

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

from panst3r.utils import batched_map

class MaskTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, ff_dim, mask_dim, num_queries, num_heads, dec_layers, lang_dim=768, normalize_before=False, num_feature_levels=3, enforce_input_project=False, two_stage=False, landscape_only=False):
        super().__init__()

        self.num_feature_levels = num_feature_levels
        if isinstance(in_dim, int):
            in_dim = [in_dim] * num_feature_levels

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.num_heads = num_heads
        self.num_layers = dec_layers
        self.landscape_only = landscape_only
        self.self_attn_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        for _ in range(dec_layers):
            self.self_attn_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.0,
                    normalize_before=normalize_before,
                )
            )

            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dropout=0.0,
                    normalize_before=normalize_before,
                )
            )

            self.ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=ff_dim,
                    dropout=0.0,
                    normalize_before=normalize_before,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        self.two_stage = two_stage
        if not self.two_stage:
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        # view embedding (to difereniate between view1 and view2)
        self.input_proj = nn.ModuleList()
        for in_d in in_dim:
            if in_d != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_d, hidden_dim, kernel_size=1))
                nn.init.xavier_normal_(self.input_proj[-1].weight)
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.lang_embed = nn.Linear(hidden_dim, lang_dim)
        self.cls_logit_scale = nn.Parameter(torch.ones([]))

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def query_selection(self, feats, pos, cls_embeddings):
        feats = torch.cat(feats, dim=0)
        pos = torch.cat(pos, dim=0)

        feats_o = self.decoder_norm(feats)
        feats_o = feats_o.transpose(0, 1)

        feats_lang = self.lang_embed(feats_o)

        # Compute cosine similarity between query and class language embeddings
        feats_lang = feats_lang / (feats_lang.norm(dim=-1, keepdim=True) + 1e-7)
        feats_class = self.cls_logit_scale.exp() * feats_lang @ cls_embeddings.unsqueeze(0).transpose(1, 2)

        topk = self.num_queries
        _, topk_idx = torch.topk(feats_class.max(-1)[0], topk, dim=1)
        topk_idx = topk_idx.T[:, :,  None].expand(-1, -1, feats.shape[-1])
        output = torch.gather(feats, 0, topk_idx)
        query_embed = torch.gather(pos, 0, topk_idx)  # TODO: use something else?

        return output, query_embed

    def get_pe_with_transpose(self, x, true_shape):
        if self.landscape_only:
            height, width = true_shape.T
            is_landscape = (width >= height).to(x.device)

            # Transpose embeddings for transposed images
            pe_land = self.pe_layer(x, None).flatten(2).permute(2, 0, 1)
            pe_port = self.pe_layer(x.transpose(-2,-1), None).flatten(2).permute(2, 0, 1)

            pos_emb = torch.where(is_landscape[None, :, None], pe_land, pe_port)
        else:
            pos_emb = self.pe_layer(x, None).flatten(2).permute(2, 0, 1)

        return pos_emb

    def forward(self, fpn_f, mask_feats, true_shape, cls_embeddings, num_memory=None, max_bs=None, deep_supervision=True, outdevice=None, multi_ar=False):
        assert len(fpn_f) == self.num_feature_levels
        if outdevice is None:
            outdevice = mask_feats.device

        if not multi_ar:
            fpn_f = [[feats] for feats in fpn_f]
            true_shape = [true_shape]

        src = [[] for _ in range(self.num_feature_levels)]
        pos = [[] for _ in range(self.num_feature_levels)]
        size_list = [[] for _ in range(self.num_feature_levels)]

        for i in range(self.num_feature_levels):
            for ar_i in range(len(true_shape)): # support for multi-ar
                B,N,_,_,_ = fpn_f[i][ar_i].shape
                N = N if num_memory is None else num_memory
                size_list[i].append(fpn_f[i][ar_i].shape[-2:])
                # Get pos emb based on view0 (all views are assumed to have the same shape)
                pos_i = self.get_pe_with_transpose(fpn_f[i][ar_i][:,0], true_shape[ar_i][:,0]) # HWxNxC
                pos[i].append(pos_i.repeat(N,1,1))
                src_i = self.input_proj[i](fpn_f[i][ar_i][:,:N]).permute(0,2,1,3,4).flatten(-3) # BxCxNHW
                src[i].append(src_i.permute(2, 0, 1) + self.level_embed.weight[i][None, None]) # NHWxNxC

        src = [torch.cat(s, dim=0) for s in src]
        pos = [torch.cat(p, dim=0) for p in pos]

        if self.two_stage:
            output, query_embed = self.query_selection(src, pos, cls_embeddings)
            outputs_class, outputs_masks, attn_mask = self.forward_prediction_heads(
                output, mask_feats, cls_embeddings, attn_mask_target_size=size_list[0],
                max_bs=max_bs, multi_ar=multi_ar)
            output = output.detach()  # do not backpropagate through proposals
        else:
            # QxNxC
            _, bs, _ = src[0].shape
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            outputs_class, outputs_masks, attn_mask = self.forward_prediction_heads(
                output, mask_feats, cls_embeddings, attn_mask_target_size=size_list[0],
                max_bs=max_bs, outdevice=outdevice, multi_ar=multi_ar)

        # Initial predictions
        predictions_class = []
        predictions_masks = []
        if deep_supervision:
            predictions_class.append(outputs_class)
            predictions_masks.append(outputs_masks)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.cross_attn_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.self_attn_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.ffn_layers[i](
                output
            )

            outputs_class, outputs_masks, attn_mask = self.forward_prediction_heads(
                output, mask_feats, cls_embeddings, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                max_bs=max_bs, outdevice=outdevice, multi_ar=multi_ar)

            if deep_supervision or i >= self.num_layers - 1:
                predictions_class.append(outputs_class)
                predictions_masks.append(outputs_masks)

        if deep_supervision:
            assert len(predictions_class) == self.num_layers + 1
        else:
            assert len(predictions_class) > 0

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_masks[-1],
            'aux_outputs': self._prepare_aux_outputs(
                predictions_class, predictions_masks
            ),
            'out_queries': output.detach()
        }
        return out

    def forward_prediction_heads(self, output, mask_feats, cls_embeddings, attn_mask_target_size=None, max_bs=None, outdevice=None, multi_ar=False):
        if outdevice is None:
            outdevice = mask_feats.device

        if not multi_ar:
            mask_feats = [mask_feats]

        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_lang = self.lang_embed(decoder_output)

        # Compute cosine similarity between query and class language embeddings
        outputs_lang = outputs_lang / (outputs_lang.norm(dim=-1, keepdim=True) + 1e-7)
        outputs_class = (self.cls_logit_scale.exp() * outputs_lang @ cls_embeddings.unsqueeze(0).transpose(1, 2)).to(outdevice)

        mask_embed = self.mask_embed(decoder_output)
        mask_embed = [mask_embed.unsqueeze(1).expand(-1, mf.shape[1], -1, -1) for mf in mask_feats]
        if attn_mask_target_size is not None:
            attn_mask_target_size = [torch.tensor(s).view(1, 1, -1).repeat(m.shape[0], m.shape[1], 1) for m, s in zip(mask_feats, attn_mask_target_size)]

        def _process_fn(mask_feats, mask_embed, attn_mask_target_size=None):
            if attn_mask_target_size is not None:
                attn_mask_target_size = attn_mask_target_size[0].tolist()

            outputs_mask, attn_mask = self._compute_masks(mask_feats.unsqueeze(1), mask_embed, attn_mask_target_size)

            if attn_mask is not None:
                return outputs_mask.squeeze(1).to(outdevice), attn_mask.squeeze(1)
            else:
                return outputs_mask.squeeze(1).to(outdevice)


        params = (mask_feats, mask_embed)
        if attn_mask_target_size is not None:
            params = (*params, attn_mask_target_size)

        outputs = batched_map(
            _process_fn,
            params,
            flatten_dims=(0, 1),
            batch_size=max_bs,
            multi_ar=True
        )

        if attn_mask_target_size is not None:
            outputs_mask, attn_mask = outputs
        else:
            outputs_mask, attn_mask = outputs, None

        if attn_mask is not None:
            attn_masks_out = []
            for msk in attn_mask: # multi-ar support
                msk = msk.permute(0,2,1,3,4).flatten(-3)
                attn_masks_out.append(msk)
                # must use bool type
                # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = torch.cat(attn_masks_out, dim=2).detach()
            attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()

        if not multi_ar:
            outputs_mask = outputs_mask[0]

        return outputs_class, outputs_mask, attn_mask

    def _compute_masks(self, mask_feats, mask_embed, attn_mask_target_size):
        outputs_mask = torch.einsum("bqc,bnchw->bnqhw", mask_embed, mask_feats)

        attn_mask = None
        if attn_mask_target_size is not None:
            B,N,Q,H,W = outputs_mask.shape
            attn_mask = outputs_mask.flatten(0,1)
            attn_mask = F.interpolate(attn_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            attn_mask = attn_mask.view(B,N,Q,*attn_mask_target_size)
        return outputs_mask, attn_mask

    @torch.jit.unused
    def _prepare_aux_outputs(self, outputs_class, outputs_seg_masks):
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be one of (relu, gelu, glu), not {activation}.")


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(-2), x.size(-1)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

