from typing import Dict, Text, Any
import laplace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch import nn
from torch import optim

from network.lightning.fc_lightning_module import FcLightningModule

import torchmetrics
from util.config import metrics as metrics_config
from network.lightning.laplace_lightning_module import LaplaceModule
from network.lightning import laplace as bayesian_laplace

def get_default_lightning_module_params() -> Dict[Text, Any]:
    return {
        "loss_function": nn.CrossEntropyLoss(),
        "optimizer": optim.Adam,
        "optimizer_params": {
            "lr": 0.001
        },
        "train_metrics": metrics_config.get_default_deterministic_train_metrics(num_classes=3),
        "val_metrics": metrics_config.get_default_deterministic_val_metrics(num_classes=3)
    }


# - Analyse der Unischerheitsmetriken: Entropy, MI, Varianz
#   - Varianz über eine bestimmte Layer analysieren -> rausfinden, welcher Teil (Channel) von dem Layer zur Unsicherheit beiträgt
#   - Erwartung: spätere Layers eine relative flache Verteilung der Varianz über alle Channels -> globale Unsicherheit, da viele Elemente schon dazu beitragen
#   -            frühere Layers, eine relative starke Verteilung der Varianz über alle Channels -> lokale Unsicherheit, lässt sich auf bestimmte Channels/Inputs zurückführen 
# - Eigenwerte der Fisher Information Matrix
#   - Eigenwerte auf einem Threshold abschneiden
#   - Analyse über Channels ? -> Durschnitt probieren
# - Schlussfolgerung zwischen beiden

def get_default_lightning_module(model: nn.Module) -> pl.LightningModule:
    params = get_default_lightning_module_params()
    return FcLightningModule(model, **params)


def get_default_lightning_trainer_params() -> Dict[Text, Any]:
    return {
        "max_epochs": 1000,
        "devices": 1,
        "enable_progress_bar": True,
        "callbacks": [
            EarlyStopping(monitor="val/loss", patience=50, mode="min", stopping_threshold=0.001),
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
