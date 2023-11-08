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

def get_default_lightning_module_params() -> Dict[Text, Any]:
    return {
        "loss_function": nn.CrossEntropyLoss(),
        "optimizer": None,
        "optimizer_params": None,
        "train_metrics": metrics_config.get_default_deterministic_train_metrics(num_classes=1000),
        "val_metrics": metrics_config.get_default_deterministic_val_metrics(num_classes=1000)
    }


def get_default_lightning_module(model: nn.Module) -> pl.LightningModule:
    params = get_default_lightning_module_params()
    return LightningClassifier(model, **params)


def get_default_lightning_trainer_params() -> Dict[Text, Any]:
    return {
        "devices": 1,
        "enable_progress_bar": True
    }

