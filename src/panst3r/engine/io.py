# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
from pathlib import Path

from croco.utils.misc import save_on_master

from panst3r import PanSt3R


def save_model(args, epoch, model: PanSt3R, optimizer, loss_scaler, fname=None):
    output_dir = Path(args.output_dir)
    if fname is None:
        fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    optim_state_dict = optimizer.state_dict()
    to_save = {
        'weights': model.state_dict(),
        'optimizer': optim_state_dict,
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)


def load_model(args, chkpt_path, model: PanSt3R, optimizer, loss_scaler):
    args.start_epoch = 0
    if chkpt_path is not None:
        checkpoint = torch.load(chkpt_path, map_location='cpu', weights_only=False)

        print("Resume checkpoint %s" % chkpt_path)
        model.load_state_dict(checkpoint['weights'], strict=False)
        args.start_epoch = checkpoint['epoch'] + 1
        optim_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(optim_state_dict)
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        else:
            print("")
        print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')


def save_final_model(args, epoch, model: PanSt3R):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'

    if args.distributed:
        assert not args.finetune_encoder

    to_save = {
        'args': args,
        'weights': model.state_dict(),
        'epoch': epoch
    }
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)
