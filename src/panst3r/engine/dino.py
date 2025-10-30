# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import numpy as np
from contextlib import nullcontext
from tqdm.auto import tqdm
from panst3r.utils import batched_map

def inference_dino(dino_encoder, imgs, true_shape_view, max_bs=None, requires_grad=False, verbose=False):
    context = nullcontext if requires_grad else torch.no_grad

    with context():
        out = batched_map(
            dino_encoder,
            (imgs, true_shape_view),
            flatten_dims=(0, 1),
            batch_size=max_bs,
            verbose=verbose,
            desc="DINO encoder",
        )

        return out

