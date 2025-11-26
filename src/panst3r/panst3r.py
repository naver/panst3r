# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import torch
from torch import nn
from contextlib import nullcontext
import numpy as np
from tqdm import tqdm

from must3r.model import *
from must3r.demo.inference import farthest_point_sampling
from panst3r.engine.retrieval import PanSt3RRetriever
from must3r.engine.inference import inference_multi_ar, stack_views

from panst3r.model import *
from panst3r.engine.must3r import inference_encoder, inference_decoder_memory, inference_decoder_render
from panst3r.engine.dino import inference_dino
from panst3r.utils import batched_map, unstack_tensors, get_dtype

class PanSt3R(nn.Module):
    def __init__(self, must3r_encoder: nn.Module, must3r_decoder: nn.Module, dino_encoder: nn.Module, panoptic_decoder: PanopticDecoder,
                 retrieval = None, preserve_gpu_mem: bool = False,
                 postprocess_default: str = 'standard', qubo_enabled: bool = False,
                 must3r_encoder_requires_grad=False, must3r_decoder_requires_grad=False, verbose: bool = False):
        super().__init__()

        self.must3r_encoder = must3r_encoder
        self.must3r_decoder = must3r_decoder
        self.dino_encoder = dino_encoder
        self.panoptic_decoder = panoptic_decoder

        self.retrieval = retrieval
        self.preserve_gpu_mem = preserve_gpu_mem
        self.verbose = verbose

        self.must3r_params = dict(
            init_num_views=2,
            batch_num_views=1,
            render_iterations=1,
        )

        self.must3r_encoder_requires_grad = must3r_encoder_requires_grad
        self.must3r_decoder_requires_grad = must3r_decoder_requires_grad

        self.postprocess_default = postprocess_default
        self.qubo_enabled = qubo_enabled

    def forward_dino(self, imgs, true_shape, max_bs=None, verbose=None):
        """DINOv2 forward pass"""

        if verbose is None:
            verbose = self.verbose

        x_dino = inference_dino(self.dino_encoder, imgs, true_shape, max_bs, requires_grad=False, verbose=verbose)
        return x_dino

    def forward_must3r_encoder(self, imgs, true_shape, max_bs=None):
        """Forward pass through the MUSt3R encoder"""

        # MUSt3R encoder
        x_must3r, pos_must3r = inference_encoder(self.must3r_encoder, imgs, true_shape, max_bs,
                                                 requires_grad=self.must3r_encoder_requires_grad, verbose=self.verbose)

        return x_must3r, pos_must3r

    def get_must3r_mem_batches(self, n_imgs):
        mem_batches = [self.must3r_params['init_num_views']]
        while (sum_b := sum(mem_batches)) != n_imgs:
            size_b = min(self.must3r_params['batch_num_views'], n_imgs - sum_b)
            mem_batches.append(size_b)
        return mem_batches

    def forward_must3r_decoder(self, x_must3r, pos_must3r, true_shape, max_bs=None):
        """Forward pass through the MUSt3R decoder"""
        mem_batches = self.get_must3r_mem_batches(x_must3r.shape[1])

        # MUSt3R decoder (first pass - memory update)
        mem, _, _ = inference_decoder_memory(self.must3r_decoder, x_must3r, pos_must3r, true_shape,
                                             mem_batches=mem_batches, requires_grad=self.must3r_decoder_requires_grad)


        # MUSt3R decoder (rendering with accumulated memory)
        # for i in range(self.must3r_params['render_iterations']):
        pointmaps, y_must3r = inference_decoder_render(self.must3r_decoder, x_must3r, pos_must3r, true_shape, mem,
                                                            max_bs=max_bs, requires_grad=self.must3r_decoder_requires_grad)

        return y_must3r, pointmaps, mem

    def _get_keyframes_retrieval(self, must3r_x, num_keyframes):
        """Gets keyframes through retrieval"""

        assert self.retrieval is not None, "Retrieval model not provided."
        assert PanSt3RRetriever is not None, "Retrieval not available. Install asmk to support retrieval."

        if not isinstance(must3r_x, list):
            must3r_x = [xi.unsqueeze(0) for xi in must3r_x.squeeze(0)]

        retriever = PanSt3RRetriever(self.retrieval, backbone=self.must3r_encoder, device=must3r_x[0].device, verbose=self.verbose)

        sim_matrix = retriever(must3r_x, device=must3r_x[0].device)
        # Cleanup
        del retriever
        torch.cuda.empty_cache()

        anchor_idx, _ = farthest_point_sampling(1-sim_matrix, N=num_keyframes, dist_thresh=None)
        sim_matrix = sim_matrix[anchor_idx, :][:, anchor_idx]

        diag = np.diag_indices(num_keyframes)
        sim_matrix[diag[0], diag[1]] = 0
        sim_sum = np.sum(sim_matrix, axis=-1)

        keyframes = [np.argmax(sim_sum)]  # start with image that has the highest overlap
        sim_matrix[:, keyframes[0]] = 0  # invalidate column
        while len(keyframes) != num_keyframes:
            # last_keyframe = keyframes[-1]
            # best_next_image = np.argmax(sim_matrix[last_keyframe])

            sim_matrix_sel = sim_matrix[np.array(keyframes)]
            best_next_image = np.unravel_index(np.argmax(sim_matrix_sel),
                                            sim_matrix_sel.shape)[1]  # we need the column index
            keyframes.append(best_next_image)
            sim_matrix[:, best_next_image] = 0

        keyframes = [anchor_idx[k] for k in keyframes]

        return keyframes

    def _forward_decoder_render(self, imgs, x_must3r, pos_must3r, true_shape, mem_must3r, mem_panst3r, classes, max_bs=None, multi_ar=False, outdevice=None):
        """Efficient forward of must3r decoder + panst3r decoder (with memory built on keyframes)"""
        device = imgs.device if not multi_ar else imgs[0].device
        if outdevice is None:
            outdevice = device

        # B, nimgs = imgs.shape[:2]
        # mem_panst3r = mem_panst3r.permute(1,0,2).unsqueeze(1).expand(-1, nimgs, -1, -1)

        def _process_slice(imgs_i, true_shape_i, x_must3r_i, pos_must3r_i):
            """Process a slice of images through the decoder."""
            x_must3r_i = x_must3r_i.to(device)
            pos_must3r_i = pos_must3r_i.to(device)

            pointmaps_i, y_must3r_i = inference_decoder_render(
                self.must3r_decoder,
                x_must3r_i.unsqueeze(1),
                pos_must3r_i.unsqueeze(1),
                true_shape_i.unsqueeze(1),
                mem=mem_must3r)

            # mem_panst3r_i = mem_panst3r_i.flatten(0,1).permute(1,0,2)

            x_dino_i = self.forward_dino(imgs_i.unsqueeze(1), true_shape_i.unsqueeze(1), max_bs, verbose=False)
            panout_i = self.panoptic_decoder((x_must3r_i.unsqueeze(1), y_must3r_i, x_dino_i), imgs_i.unsqueeze(1),
                                             pos_must3r_i.unsqueeze(1), true_shape_i.unsqueeze(1), classes,
                                             max_bs=max_bs, outdevice=outdevice, memory_queries=mem_panst3r)

            return pointmaps_i.squeeze(1).to(outdevice), panout_i['pred_masks'].squeeze(1)

        pointmaps, pan_masks = batched_map(
            _process_slice,
            (imgs, true_shape, x_must3r, pos_must3r),
            batch_size=max_bs,
            flatten_dims=(0, 1),
            verbose=self.verbose,
            desc="Rendering and decoding remaining views",
            multi_ar=multi_ar
        )

        return pointmaps, pan_masks

    def forward_inference_multi_ar(self, imgs, true_shape, classes, num_keyframes=None,
                                   use_retrieval=False, max_bs=None, outdevice=None, amp=False):
        """Inference forward function for PanSt3R. Supports retrieval, and memory-efficient processing (in chunks) and multi-aspect ratio inputs."""
        dtype = get_dtype(amp)
        device = imgs[0].device
        with torch.autocast(device.type, dtype=dtype):
            x_must3r, pos_must3r = self.forward_must3r_encoder(imgs, true_shape, max_bs=max_bs)

        # Select keyframes (retrieval/linspace/all)
        N = len(imgs)
        if use_retrieval:
            keyframes = self._get_keyframes_retrieval([xi.unsqueeze(0).float() for xi in x_must3r], num_keyframes)
        else:
            # Use first N images as keyframes
            if num_keyframes is None or num_keyframes > N:
                keyframes = list(range(N))
            else:
                keyframes = np.linspace(0, N - 1, num_keyframes, dtype=int).tolist()

        not_keyframes = sorted(set(range(N)).difference(set(keyframes)))
        assert (len(keyframes) + len(not_keyframes)) == N

        # reorder images
        new_idx = keyframes + not_keyframes
        imgs = [imgs[i] for i in new_idx]
        true_shape = true_shape[new_idx]
        x_must3r = [x_must3r[i] for i in new_idx]
        pos_must3r = [pos_must3r[i] for i in new_idx]

        # Step 1: Run decoder on keyframes
        imgs_kf = imgs[:num_keyframes]
        true_shape_kf = list(true_shape[:num_keyframes].unbind(0))
        x_must3r_kf = x_must3r[:num_keyframes]
        pos_must3r_kf = pos_must3r[:num_keyframes]

        with torch.autocast(device.type, dtype=dtype):
            mem_kf, _, _ = inference_multi_ar(self.must3r_encoder, self.must3r_decoder, imgs_kf,
                                            img_ids=keyframes, true_shape=true_shape_kf,
                                            mem_batches=self.get_must3r_mem_batches(num_keyframes),
                                            max_bs=max_bs, to_render=[],
                                            encoder_precomputed_features=(x_must3r_kf, pos_must3r_kf),
                                            preserve_gpu_mem=True, return_mem=True, num_refinements_iterations=0)

            true_shape_kf, index_stacks, x_must3r_kf, pos_must3r_kf, imgs_kf = stack_views(true_shape[:num_keyframes],
                                                                                        [x_must3r_kf,
                                                                                            pos_must3r_kf,
                                                                                            imgs_kf],
                                                                                        max_bs=max_bs)
            pbar = tqdm(zip(x_must3r_kf, pos_must3r_kf, true_shape_kf, imgs_kf), total=len(x_must3r_kf))
            pointmaps_kf = []
            feats_must3r_kf = []
            x_dino_kf = []
            for x_stack, pos_stack, true_shape_stack, imgs_stack in pbar:
                pointmaps_stack, y_must3r = inference_decoder_render(self.must3r_decoder,
                                                                    x_stack.unsqueeze(0).to(device),
                                                                    pos_stack.unsqueeze(0).to(device),
                                                                    true_shape_stack.unsqueeze(0),
                                                                    mem_kf, max_bs=max_bs,
                                                                    requires_grad=self.must3r_decoder_requires_grad)

                x_dino = self.forward_dino(imgs_stack.unsqueeze(0).to(device), true_shape_stack.unsqueeze(0),
                                            max_bs=max_bs, verbose=False)

                pointmaps_kf.append(pointmaps_stack.to(outdevice))
                feats_must3r_kf.append(y_must3r)
                x_dino_kf.append(x_dino)

        # TODO: check how well panoptic_decoder runs with autocast
        # Batch dim expected by decoder
        x_must3r_kf = [x.unsqueeze(0).to(device) for x in x_must3r_kf]
        imgs_kf = [x.unsqueeze(0).to(device) for x in imgs_kf]
        pos_must3r_kf = [x.unsqueeze(0).to(device) for x in pos_must3r_kf]
        true_shape_kf = [x.unsqueeze(0).to(device) for x in true_shape_kf]

        pan_feats_kf = (x_must3r_kf, feats_must3r_kf, x_dino_kf)
        panout_kf = self.panoptic_decoder(pan_feats_kf, imgs_kf, pos_must3r_kf, true_shape_kf, classes,
                                          outdevice=outdevice, max_bs=max_bs, multi_ar=True)

        panout = {
            'pred_logits': panout_kf['pred_logits'],
            'pred_masks': unstack_tensors(index_stacks, panout_kf['pred_masks']),
            'out_queries': panout_kf['out_queries']
        }
        pointmaps = unstack_tensors(index_stacks, pointmaps_kf)

        if len(not_keyframes) > 0:
            # # Step 2: Run decoder on remaining views (render-only)
            torch.cuda.empty_cache()
            true_shape_nkf, index_stacks, x_must3r_nkf, pos_must3r_nkf, imgs_nkf = stack_views(true_shape[num_keyframes:],
                                                                                               [x_must3r[num_keyframes:],
                                                                                                pos_must3r[num_keyframes:],
                                                                                                imgs[num_keyframes:]],
                                                                                               max_bs=max_bs)
            true_shape_nkf = [x.unsqueeze(0) for x in true_shape_nkf]
            x_must3r_nkf = [x.unsqueeze(0) for x in x_must3r_nkf]
            pos_must3r_nkf = [x.unsqueeze(0) for x in pos_must3r_nkf]
            imgs_nkf = [x.unsqueeze(0) for x in imgs_nkf]


            pointmaps_nkf, pan_masks_nkf = self._forward_decoder_render(imgs_nkf, x_must3r_nkf, pos_must3r_nkf, true_shape_nkf,
                                                                        mem_must3r=mem_kf, mem_panst3r=panout_kf['out_queries'],
                                                                        classes=classes, max_bs=max_bs, multi_ar=True, outdevice=outdevice)

            pan_masks_nkf = unstack_tensors(index_stacks, pan_masks_nkf)
            pointmaps_nkf = unstack_tensors(index_stacks, pointmaps_nkf)

            # # Step 3: Combine predictions
            panout['pred_masks'] = panout['pred_masks'] + pan_masks_nkf
            pointmaps = pointmaps + pointmaps_nkf

        # Order predictions back to original order
        inv_idx = np.argsort(new_idx)
        pointmaps = [pointmaps[i] for i in inv_idx]
        panout['pred_masks'] = [panout['pred_masks'][i] for i in inv_idx]

        return pointmaps, panout

    def forward(self, imgs, true_shape, classes, max_bs=None, outdevice=None):

        x_dino = self.forward_dino(imgs, true_shape, max_bs)
        x_must3r, pos_must3r = self.forward_must3r_encoder(imgs, true_shape, max_bs)
        y_must3r, pointmaps, _ = self.forward_must3r_decoder(x_must3r, pos_must3r, true_shape, max_bs)

        pan_feats = (x_must3r, y_must3r, x_dino)

        panout = self.panoptic_decoder(pan_feats, imgs, pos_must3r, true_shape, classes, outdevice=outdevice)

        return panout, pointmaps

    def set_vocab(self, class_names, device=None):
        self.panoptic_decoder.text_encoder.set_vocab(class_names, device=device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, retrieval_path=None):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        assert 'args' in ckpt, "Checkpoint must contain 'args' with model parameters."

        must3r_encoder = eval(ckpt['args'].must3r_encoder)
        must3r_decoder = eval(ckpt['args'].must3r_decoder)
        dino_encoder = eval(ckpt['args'].dino_encoder)
        panoptic_decoder = eval(ckpt['args'].panoptic_decoder)
        retrieval = ckpt['retrieval'] if 'retrieval' in ckpt else None

        panst3r = PanSt3R(
            must3r_encoder=must3r_encoder,
            must3r_decoder=must3r_decoder,
            dino_encoder=dino_encoder,
            panoptic_decoder=panoptic_decoder,
            retrieval=retrieval,
            postprocess_default=ckpt['args'].postprocess_default,
            qubo_enabled=ckpt['args'].qubo_enabled,
        )

        panst3r.load_state_dict(ckpt['weights'], strict=False)

        return panst3r
