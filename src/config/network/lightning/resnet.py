from math import gamma
from typing import Dict, Text, Any
import laplace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch import nn
from torch import optim

from network.lightning.classifier import LightningClassifier

import torchmetrics
from config import metrics as metrics_config
from network.lightning.laplace import LaplaceModule
from network.lightning import laplace as bayesian_laplace

# Hyperparameters from https://github.com/akamaster/pytorch_resnet_cifar10, except for lf_scheduler
def get_default_lightning_module_params() -> Dict[Text, Any]:
    return {
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': optim.SGD,
        "optimizer_params": {
            # "lr": 5e-2, # Higher for dropouts 
            # "lr": 1, 
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4
        },
        "lr_scheduler": optim.lr_scheduler.ReduceLROnPlateau,
        # "lr_scheduler": optim.lr_scheduler.MultiStepLR,
        "lr_scheduler_params": {
            'mode': 'min',
            # 'factor': 0.5,
            # 'patience': 5,
            # 'milestones': [60, 100, 120, 160], 
            # 'milestones': [100, 150], 
            # 'gamma': 0.5,
            # 'initial_lr': 1e-1,
        },
        "train_metrics": metrics_config.get_default_deterministic_train_metrics(num_classes=10),
        "val_metrics": metrics_config.get_default_deterministic_val_metrics(num_classes=10)
    }


def get_default_lightning_module(model: nn.Module) -> pl.LightningModule:
    params = get_default_lightning_module_params()
    return LightningClassifier(model, **params)


def get_default_lightning_trainer_params() -> Dict[Text, Any]:
    return {
        "max_epochs": 200,
        "devices": 1,
        "enable_progress_bar": True,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 1,
        "callbacks": [
            EarlyStopping(monitor="val/loss", patience=200, mode="min"),
            ModelCheckpoint(
                filename='epoch={epoch}-step={step}-val_loss={val/loss:.2f}-best',
                save_top_k=1,
                monitor='val/loss',
                mode='min',
                auto_insert_metric_name=False,
                verbose=True,
            )
        ]
    }

