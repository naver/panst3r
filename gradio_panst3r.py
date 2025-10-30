# Copyright (C) 2025-present Naver Corporation. All rights reserved.
#
# --------------------------------------------------------
# dust3r gradio demo executable
# --------------------------------------------------------
import os
import torch
import tempfile

from must3r.model import *
from must3r.model.blocks.attention import toggle_memory_efficient_attention, has_xformers
from tools.demo_panst3r import get_args_parser, main_demo

from panst3r import PanSt3R

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    toggle_memory_efficient_attention(enabled=has_xformers)

    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    weights_path = args.weights
    panst3r = PanSt3R.from_checkpoint(weights_path, args.retrieval)
    panst3r.verbose = True
    panst3r.panoptic_decoder.text_encoder.change_mode(fixed_vocab=False)
    panst3r = panst3r.eval().to(args.device)

    with tempfile.TemporaryDirectory(suffix='panst3r_gradio_demo') as tmpdirname:
        if args.verbose:
            print('Outputing stuff in', tmpdirname)
        main_demo(tmpdirname, panst3r, args.device, args.image_size, server_name, args.server_port, args.viser_port,
                  verbose=args.verbose, allow_local_files=args.allow_local_files, camera_animation=args.camera_animation, amp=args.amp)
