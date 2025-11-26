# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import os
import torch
from must3r.demo.inference import MUSt3R_Retriever
from must3r.retrieval.model import RetrievalModel

from asmk.asmk_method import ASMKMethod
from asmk.codebook import Codebook
from asmk.index import initialize_index

class PanSt3RRetriever(MUSt3R_Retriever):
    def __init__(self, ckpt, backbone, device='cuda', verbose=True):
        # If file load the checkpoint.
        if isinstance(ckpt, str) and os.path.isfile(ckpt):
            ckpt = torch.load(ckpt, 'cpu', weights_only=False)
        assert backbone is not None
        ckpt_args = ckpt['args']
        self.model = RetrievalModel(
            backbone, freeze_backbone=ckpt_args.freeze_backbone, prewhiten=ckpt_args.prewhiten,
            hdims=list(map(int, ckpt_args.hdims.split('_'))) if len(ckpt_args.hdims) > 0 else "",
            residual=getattr(ckpt_args, 'residual', False), postwhiten=ckpt_args.postwhiten,
            featweights=ckpt_args.featweights, nfeat=ckpt_args.nfeat
        ).to(device)
        self.device = device
        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        assert all(k.startswith('backbone') for k in msg.missing_keys)
        assert len(msg.unexpected_keys) == 0
        self.imsize = ckpt_args.imsize

        asmk_codebook = ckpt['asmk_codebook']
        asmk_params = ckpt['asmk_params']

        # load the asmk codebook
        device = torch.device(device)
        gpu_id = None
        if device.type == 'cuda':
            if device.index is None:
                gpu_id = torch.cuda.current_device()
            else:
                gpu_id = device.index

        index_factory = initialize_index(gpu_id)
        cdb = Codebook.initialize_from_state(asmk_codebook, index_factory=index_factory)
        cdb.index()

        self.asmk = ASMKMethod(asmk_params, {}, codebook=cdb)
