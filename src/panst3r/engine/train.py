# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import sys
import math
from pathlib import Path
from typing import List, Sized
from itertools import chain

import numpy as np
import torch
from torch import nn

import croco.utils.misc as dist
# from must3r.engine.train import select_batch
from panst3r import PanSt3R


def select_batch(device, args, rng, memory_num_views, progress, imgs, true_shape, nimgs):
    to_skip = 0
    to_render = None

    # process it dust3r like, one image at a time, except for initialization
    mem_batches = [memory_num_views]

    return imgs, true_shape, memory_num_views, to_skip, to_render, to_skip, mem_batches

def loss_of_one_batch(args, batch: List[dict], model: PanSt3R, rng: np.random.Generator,
                      criterion: nn.Module, device: torch.device, progress: float = 1, dtype: torch.dtype = torch.float32, classes: List[str] = None, ret: str = None):

    imgs = [b['img'] for b in batch]
    imgs = torch.stack(imgs, dim=1).to(device)  # B, nimgs, 3, H, W
    nimgs = imgs.shape[1]

    true_shape = [b['true_shape'] for b in batch]
    true_shape = torch.stack(true_shape, dim=1).to(device)  # B, nimgs, 3, H, W

    with torch.amp.autocast("cuda", dtype=dtype):
        panout, x_out = model(imgs, true_shape, classes, max_bs=args.max_batch_size)

    with torch.amp.autocast("cuda", dtype=torch.float32):
        loss, loss_details = criterion(batch, panout, classes)

    result = dict(x_out=x_out, loss=(loss, loss_details), panout=panout)

    return result[ret] if ret else result


def get_dtype(args):
    if args.amp:
        dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    else:
        dtype = torch.float32
    return dtype

def unwrap(m):
    # handles nested wrappers (DDP, DataParallel, etc.)
    return m.module if hasattr(m, "module") else m

def train_one_epoch(model: PanSt3R, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    loss_scaler, args,
                    log_writer=None):

    model.train()

    metric_logger = dist.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    # fix the seed
    seed = args.seed + epoch * dist.get_world_size() + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed=args.seed + epoch)

    # get params
    params = [unwrap(model).panoptic_decoder.parameters()]
    if args.finetune_must3r_encoder:
        params.append(unwrap(model).must3r_encoder.parameters())
    if args.finetune_must3r_decoder:
        params.append(unwrap(model).must3r_decoder.parameters())
    params = list(chain(*params))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        progress = epoch_f / args.epochs

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            dist.adjust_learning_rate(optimizer, epoch_f, args)

        classes = data_loader.dataset.classes

        result = loss_of_one_batch(
            args,
            batch,
            model=model,
            rng=rng,
            criterion=criterion,
            progress=progress,
            device=device,
            dtype=args.dtype,
            classes=classes,
        )

        loss, loss_details = result['loss']
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter

        loss_scaler(loss, optimizer, parameters=params,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = dist.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            log_vals = {
                'train/loss': loss_value_reduce,
                'train/lr': lr,
                'train/iter': epoch_f,
            }
            for name, val in loss_details.items():
                log_vals['train/' + name] = val

            log_writer.log(log_vals, epoch_f)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
