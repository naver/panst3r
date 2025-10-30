# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import os
import os.path as osp
from torch import Tensor
import torch.distributed as dist
import torch.utils.tensorboard as tb
from PIL import Image
from pathlib import Path

try: import wandb
except ImportError: wandb = None

try: import mlflow
except ImportError: mlflow = None

__all__ = ['Logger', 'TBLogger', 'WandbLogger', 'MLFlowLogger']

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"] if "MLFLOW_TRACKING_URI" in os.environ else None

class Logger:
    def __init__(self):
        pass

    def log(self, metric_dict: dict[str, float], epoch_f: float):
        """Log metrics to the logger.

        Args:
            metric_dict (dict[str, float]): Dictionary of name: value metrics to log.
            epoch_f (float): Epoch as a float.
        """
        raise NotImplementedError()

    def log_images(self, images: dict[str, Tensor], epoch_f: float):
        """Log images to the logger.

        Args:
            images (dict[str, torch.Tensor]): Dictionary of name: image tensor pairs.
            epoch_f (float): Epoch as a float.
        """
        raise NotImplementedError()

    def log_config(self, config_dict):
        raise NotImplementedError()

    def flush(self):
        pass


class TBLogger(Logger):
    def __init__(self, log_dir: str, **kwargs):
        super().__init__()
        self.log_dir = log_dir
        self.writer = tb.SummaryWriter(log_dir, **kwargs)

    def log(self, metric_dict: dict[str, float], epoch_f: float):
        # Disretize continuous epoch value to integer step (epoch every 1000 steps)
        epoch_int = int(epoch_f*1000)
        for name, value in metric_dict.items():
            self.writer.add_scalar(name, value, epoch_int)

    def log_images(self, images: dict[str, Tensor], epoch_f: float):
        print("WARN: TBLogger does not currently support logging images")

    def flush(self):
        return self.writer.flush()

    def __repr__(self):
        return f'TBLogger({self.log_dir})'

class WandbLogger(Logger):
    def __init__(self, log_dir: str, project: str = "3drecon", config: dict = None, **kwargs):
        super().__init__()

        assert wandb is not None, "wandb not installed (pip install wandb)"

        self.log_dir = log_dir
        self.project = project
        wandb.init(name=osp.basename(log_dir), project=project, dir=log_dir, config=config, **kwargs)

    def log(self, metric_dict: dict[str, float], epoch_f: float):
        # Disretize continuous epoch value to integer step (epoch every 1000 steps)
        metric_dict = metric_dict.copy()
        metric_dict['epoch'] = epoch_f
        wandb.log(metric_dict)

    def log_images(self, images: dict[str, Tensor], epoch_f: float):
        log_dict = {'epoch': epoch_f}
        for name, img in images.items():
            log_dict[name] = wandb.Image(img)
        wandb.log(log_dict)

    def __repr__(self):
        return f'WandbLogger({self.log_dir}, {self.project})'


class MLFlowLogger(Logger):
    def __init__(self, log_dir: str, project: str = "3drecon", config: dict = None, **kwargs):
        super().__init__()

        assert mlflow is not None, "mlflow not installed (pip install mlflow)"

        self.log_dir = Path(log_dir)
        self.project = project

        if MLFLOW_TRACKING_URI is not None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(project)
        self.run = mlflow.start_run(run_name=self.log_dir.name)

        if config:
            for key, value in config.items():
                mlflow.log_param(key, value)

    def log(self, metric_dict: dict[str, float], epoch_f: float):
        assert mlflow is not None, "mlflow not installed (pip install mlflow)"

        metric_dict['epoch'] = epoch_f
        mlflow.log_metrics(metric_dict, step=int(1000 * epoch_f))

    def log_images(self, images: dict[str, Tensor], epoch_f: float):
        assert mlflow is not None, "mlflow not installed (pip install mlflow)"

        for name, img in images.items():
            mlflow.log_image(img, key=name, step=int(epoch_f))

    def __repr__(self):
        return f"MLFlowLogger({self.log_dir}, {self.project})"

    def __del__(self):
        assert mlflow is not None, "mlflow not installed (pip install mlflow)"

        if mlflow.active_run():
            mlflow.end_run()

class LoggerList(Logger):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def log(self, metric_dict: dict[str, float], epoch_f: float):
        for logger in self.loggers:
            logger.log(metric_dict, epoch_f)

    def log_images(self, images: dict[str, Tensor], epoch_f: float):
        for logger in self.loggers:
            logger.log_images(images, epoch_f)

    def __repr__(self):
        return f'LoggerList({self.loggers})'
