from typing import Dict, Text, Any
import laplace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch import nn
from torch import optim
from config.mode import ModelMode

from network.lightning.classifier import LightningClassifier

import torchmetrics
from config import metrics as metrics_config
from network.lightning.laplace import LaplaceModule
from network.lightning import laplace as bayesian_laplace

def get_default_lightning_module_params() -> Dict[Text, Any]:
    return {
        'loss_function': nn.CrossEntropyLoss(),
        'optimizer': optim.Adam,
        "optimizer_params": {},
        "train_metrics": metrics_config.get_default_deterministic_train_metrics(num_classes=10),
        "val_metrics": metrics_config.get_default_deterministic_val_metrics(num_classes=10)
    }


def get_default_lightning_module(model: nn.Module) -> pl.LightningModule:
    params = get_default_lightning_module_params()
    return LightningClassifier(model, **params)


def get_default_lightning_trainer_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "lenet5":
        max_epochs = 10
    elif model_mode == "cifar10_lenet5":
        max_epochs = 30
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    return {
        "max_epochs": max_epochs,
        "devices": 1,
        "enable_progress_bar": True,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 1,
        "callbacks": [
            EarlyStopping(monitor="val/loss", patience=3, mode="min"),
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

