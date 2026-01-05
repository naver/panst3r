# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import argparse
from argparse import Namespace
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import uuid
import logging

import torch
import torch.backends.cudnn as cudnn

from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler
import croco.utils.misc as dist

from dust3r.datasets import *

from must3r.model import *
import must3r.engine.optimizer as optim
from must3r.engine.train import build_dataset
from must3r.model.blocks.attention import toggle_memory_efficient_attention

from panst3r import PanSt3R
import panst3r.engine.train as training
import panst3r.engine.io as io
from panst3r.logging import *
from panst3r.datasets import *
from panst3r.model import *
from panst3r.criterion import *


LOGGING_PROJECT = 'panst3r'

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

def get_resolution(res_list):
    nb_res = len(res_list)
    if nb_res > 1:
        resolution=[]
        for i in range(nb_res):
            hwres=(res_list[i][0], res_list[i][1])
            resolution.append(hwres)
    else:
        resolution=(res_list[0][0], res_list[0][1])
    return resolution

def parse_obj_config(cfg_node):
    """Simple parser for a DictConfig object to a string representing the object instantiation."""

    if isinstance(cfg_node, str):
        return f"'{cfg_node}'"

    if not isinstance(cfg_node, DictConfig) or '_target_' not in cfg_node:
        return cfg_node

    cls_name = cfg_node._target_
    args = {k: parse_obj_config(v) for k, v in cfg_node.items() if k != '_target_'}

    return f"{cls_name}({', '.join([f'{k}={v}' for k, v in args.items()])})"

def args_from_cfg(cfg):
    myargs={}

    myargs["must3r_encoder"] = parse_obj_config(cfg.model.must3r_encoder)
    myargs["must3r_decoder"] = parse_obj_config(cfg.model.must3r_decoder)
    myargs["dino_encoder"] = parse_obj_config(cfg.model.dino_encoder)
    myargs["panoptic_decoder"] = parse_obj_config(cfg.model.panoptic_decoder)
    myargs["criterion"] = parse_obj_config(cfg.training.criterion)

    num_views=cfg.data.db_options.num_views
    min_memory_num_views=cfg.data.db_options.min_memory_num_views
    max_memory_num_views=cfg.data.db_options.max_memory_num_views
    aug_crop=cfg.data.db_options.aug_crop

    db_options = f"{num_views=}, {min_memory_num_views=}, {max_memory_num_views=}, {aug_crop=}"


    myargs["dataset"] = ""
    nb_trainset=len(cfg.data.dataset)
    resolution=get_resolution(cfg.data.train_options.resolution)
    train_options = db_options + f", {resolution=}, transform={cfg.data.train_options.transform}"
    for i in range(nb_trainset):
        dataset_name=cfg.data.dataset[i]
        ROOT=cfg.data[dataset_name].train_root
        if cfg.data[dataset_name].trainsplit is not None:
            split=cfg.data[dataset_name].trainsplit
            myargs["dataset"] += f"{cfg.data.train_options.ds_size} @ {dataset_name}({ROOT=}, {split=}, {train_options}, )"
        else:
            myargs["dataset"] += f"{cfg.data.train_options.ds_size} @ {dataset_name}({ROOT=}, {train_options})"
        if i < nb_trainset-1:
            myargs["dataset"] += " + "


    resolution=get_resolution(cfg.data.test_options.resolution)

    for k, v in cfg.training.items():
        if k not in myargs:
            myargs[k] = v

    return Namespace(**myargs)

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    assert os.environ.get('MKL_NUM_THREADS') == '1', 'otherwise inefficient'
    assert os.environ.get('NUMEXPR_NUM_THREADS') == '1', 'otherwise inefficient'
    assert os.environ.get('OMP_NUM_THREADS') == '1', 'otherwise inefficient'

    if cfg.exp.resume:
        assert Path(cfg.exp.exp_dir).exists(), f"Resuming failed: {cfg.exp.exp_dir} does not exist."
        print(f"Resuming from {cfg.exp.exp_dir}")
        cfg = OmegaConf.load(Path(cfg.exp.exp_dir) / 'config.yaml')
        cfg.exp.resume = True
    else:
        if not cfg.exp.run_id :
            run_id = uuid.uuid4().hex
            with open_dict(cfg):
                cfg.exp.run_id = run_id

    #breakpoint()

    args = args_from_cfg(cfg)
    print(args)

    dist.init_distributed_mode(args)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.backends.cuda.matmul.allow_tf32 = not args.disable_tf32
    torch.backends.cudnn.allow_tf32 = not args.disable_tf32

    toggle_memory_efficient_attention(enabled=args.use_memory_efficient_attention)

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    last_ckpt_fname = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.dataset))
    start_time = time.time()
    data_loader_train = build_dataset(args, eval(args.dataset))

    print('Loading MUSt3R encoder: {:s}'.format(args.must3r_encoder))
    must3r_encoder = eval(args.must3r_encoder)
    print('Loading MUSt3R decoder: {:s}'.format(args.must3r_decoder))
    must3r_decoder = eval(args.must3r_decoder)
    print('Building DINO: {:s}'.format(args.dino_encoder))
    dino_encoder = eval(args.dino_encoder)
    print('Building panoptic decoder: {:s}'.format(args.panoptic_decoder))
    panoptic_decoder = eval(args.panoptic_decoder)

    model = PanSt3R(
        must3r_encoder=must3r_encoder,
        must3r_decoder=must3r_decoder,
        dino_encoder=dino_encoder,
        panoptic_decoder=panoptic_decoder,
        must3r_encoder_requires_grad=args.finetune_must3r_encoder,
        must3r_decoder_requires_grad=args.finetune_must3r_decoder
    )
    model.to(device)

    # Initialize the text encoder with all possible classes
    class_names = data_loader_train.dataset.classes
    class_names = list(set(class_names))

    model.set_vocab(class_names, device)

    criterion = eval(args.criterion)

    args.dtype = training.get_dtype(args)

    assert criterion.label_mode == model.panoptic_decoder.label_mode, 'Panoptic decoder and criterion must have same label_mode'


    if args.must3r_chkpt and last_ckpt_fname is None:
        print('Loading MUSt3R weights: ', args.must3r_chkpt)
        ckpt = torch.load(args.must3r_chkpt, map_location=device, weights_only=False)
        print(model.must3r_encoder.load_state_dict(ckpt['encoder'], strict=False))
        print(model.must3r_decoder.load_state_dict(ckpt['decoder'], strict=False))
        del ckpt

    if args.chkpt and last_ckpt_fname is None:
        print('Loading pretrained: ', args.chkpt)
        ckpt = torch.load(args.chkpt, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt['weights'], strict=False))
        del ckpt  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * dist.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Freeze parameters
    model.dino_encoder.requires_grad_(False)
    model.must3r_encoder.requires_grad_(args.finetune_must3r_encoder)
    model.must3r_decoder.requires_grad_(args.finetune_must3r_decoder)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False, static_graph=False,
            broadcast_buffers='linearconv' not in args.must3r_decoder.lower())
        model_without_ddp = model.module

    param_groups = optim.get_parameter_groups(model_without_ddp, 0, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats):
        if dist.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname):
        io.save_model(args=args, model=model_without_ddp,
                      optimizer=optimizer, loss_scaler=loss_scaler,
                      epoch=epoch, fname=fname)

    io.load_model(args=args, chkpt_path=last_ckpt_fname, model=model_without_ddp,
                  optimizer=optimizer, loss_scaler=loss_scaler)

    log_writer = None
    if global_rank == 0 and args.output_dir is not None:
        log_writer = TBLogger(args.output_dir, purge_step=args.start_epoch*1000)
        if args.logger == 'wandb':
            log_writer = WandbLogger(args.output_dir, project=LOGGING_PROJECT, config=vars(args))
        elif args.logger == 'mlflow':
            log_writer = MLFlowLogger(args.output_dir, project=LOGGING_PROJECT, config=vars(args))
        elif args.logger != 'tensorboard':
            print("WARN: Invalid logger selected, defaulting to tensorboard.")

    # if args.vis_dataset is not None:
    #     visualizer = PanopticVisualizer()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_stats = training.train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=args, log_writer=log_writer,
        )

        write_log_stats(epoch, train_stats)

        # Save the 'last' checkpoint
        if epoch >= args.start_epoch:
            save_model(epoch, 'last')
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch, str(epoch))

        # Test on multiple datasets
        if (epoch >= args.start_epoch and
                args.eval_every > 0 and epoch % args.eval_every == 0):
            print("TODO: testing would be done here...")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    io.save_final_model(args, args.epochs, model_without_ddp)



if __name__ == '__main__':
   main()
