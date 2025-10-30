# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import numpy as np
from contextlib import nullcontext

from panst3r.utils import batched_map
from must3r.engine.inference import encoder_multi_ar

def inference_encoder(encoder, imgs, true_shape, max_bs=None, requires_grad=False, verbose=False):
    context = nullcontext if requires_grad else torch.no_grad

    with context():
        if isinstance(imgs, list):  # multiple image size
            x, pos = encoder_multi_ar(encoder, imgs, true_shape, verbose=verbose, max_bs=max_bs,
                                      device=true_shape.device, preserve_gpu_mem=True)
        else:
            x, pos = batched_map(
                encoder,
                (imgs, true_shape),
                batch_size=max_bs,
                flatten_dims=(0, 1),
                verbose=verbose,
                desc="MUSt3R Encoder"
            )

    return x, pos

def inference_decoder_memory(decoder, x, pos, true_shape, mem_batches, requires_grad=False):
    mem_batches = [0] + np.cumsum(mem_batches).tolist()
    mem_reduction = [None for _ in range(len(mem_batches))] # TODO: remove

    mem = None
    outshape = None

    B = x.shape[0]
    context = nullcontext if requires_grad else torch.no_grad
    with context():
        pointmaps_0 = []
        decout = []
        for i in range(len(mem_batches) - 1):
            xi = x[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
            posi = pos[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
            true_shapei = true_shape[:, mem_batches[i]:mem_batches[i + 1]].contiguous()

            mem, pointmaps_0i, feats_i = decoder(xi, posi, true_shapei, mem, render=False, return_feats=True)
            decout_i = feats_i[-1]
            if outshape is None:
                outshape = pointmaps_0i.shape

            # if self.preserve_gpu_mem:
            #     torch.cuda.empty_cache()

            pointmaps_0.append(pointmaps_0i)
            decout.append(decout_i)


        # concatenate the first pass pointmaps together
        if len(pointmaps_0) > 0:
            # B, mem_batches[-1] - mem_batches[train_decoder_skip], N, D
            pointmaps_0 = torch.concatenate(pointmaps_0, dim=1)
            decout = torch.cat(decout, dim=1)
        else:
            pointmaps_0 = torch.empty((B, 0, *outshape[2:]), dtype=x.dtype, device=x.device)
            decout = None

    # TODO: if you want to do multiple passes of memory building, pass previous memory and do memory
    # update like: https://github.com/naver/must3r/blob/b2f4d981a0f7855987bf15e1a7baa2a65c39b9ba/must3r/engine/inference.py#L426-L445

    return mem, pointmaps_0, decout

def inference_decoder_render(decoder, x, pos, true_shape, mem, to_render=None, max_bs=None, requires_grad=False):
    B, nimgs, N, D = x.shape

    context = nullcontext if requires_grad else torch.no_grad
    with context():
        mem_vals, mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = mem
        try:
            _, Nmem, Dmem = mem_vals[-1].shape
        except Exception as e:
            _, Nmem, Dmem = mem_vals[0][-1].shape


        if max_bs is None or B * nimgs <= max_bs:
            # with to_render, you can select a list of images to render, instead of rendering all of them
            if to_render is not None:
                x = x[:, to_render].contiguous()
                x_dino = x_dino[:, to_render].contiguous()
                pos = pos[:, to_render].contiguous()
                true_shape = true_shape[:, to_render].contiguous()
                nimgs = x.shape[1]

            # render all images (concat them in the batch dimension for efficiency)
            _, pointmaps, feats = decoder(x, pos, true_shape, mem, render=True, return_feats=True)
            decout = feats[-1]
        else:
            # can also do it slice by slice in case all images don't fit at once
            x_view = x.view(B * nimgs, N, D)
            pos_view = pos.view(B * nimgs, N, 2)
            true_shape_view = true_shape.view(B * nimgs, 2)

            pointmaps = []
            decout = []

            mem_vals = [mem_vals[i].unsqueeze(1).expand(B, nimgs, Nmem, Dmem).reshape(B * nimgs, Nmem, Dmem)
                        for i in range(len(mem_vals))]
            mem_vals_splits = [torch.split(mem_vals[i], max_bs) for i in range(len(mem_vals))]
            mem_labels = mem_labels.unsqueeze(1).expand(B, nimgs, Nmem).reshape(B * nimgs, Nmem)
            mem_labels_splits = torch.split(mem_labels, max_bs)

            for lidx, (x_view_slice, pos_view_slice, true_shape_view_slice) in enumerate(
                zip(torch.split(x_view, max_bs),
                    torch.split(pos_view, max_bs),
                    torch.split(true_shape_view, max_bs))):
                memi = [m[lidx] for m in mem_vals_splits]
                mem_labelsi = mem_labels_splits[lidx]
                mem_i, xi_out, feats_i = decoder(x_view_slice.unsqueeze(1),
                                    pos_view_slice.unsqueeze(1),
                                    true_shape_view_slice.unsqueeze(1),
                                    (memi, mem_labelsi, mem_nimgs, mem_protected_imgs, mem_protected_tokens),
                                    render=True, return_feats=True)
                decout_i = feats_i[-1]
                pointmaps.append(xi_out.squeeze(1))
                decout.append(decout_i.squeeze(1))
            pointmaps = torch.concatenate(pointmaps)
            pointmaps = pointmaps.view(B, nimgs, *pointmaps.shape[1:])
            decout = torch.concatenate(decout)
            decout = decout.view(B, nimgs, *decout.shape[1:])

    return pointmaps, decout
