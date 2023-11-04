from typing import Literal, Optional, Dict, Text

from laplace import ParametricLaplace

import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics

import laplace as laplace_lib
from laplace import utils as laplace_libutils
import laplace.utils.matrix
from network.bayesian.mc_dropout import DropoutHook

from util import verification, assertion, device as device_util

PredictionMode = Literal["deterministic", "bayesian"]

class McDropoutModule(pl.LightningModule):
    def __init__(self, dropout_hook: DropoutHook, prediction_mode: PredictionMode = "deterministic", n_samples: int = 1, val_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None):
        super().__init__()
        verification.check_not_none(dropout_hook)
        verification.check_not_none(dropout_hook.model)
        self.dropout_hook = dropout_hook
        # self.dropout_hook.model.to(self._device)
        self.configure_prediction_mode(prediction_mode, n_samples)
        self.val_metrics = nn.ModuleDict(val_metrics if val_metrics is not None else {})
        self.save_hyperparameters(ignore="dropout_hook")


    def configure_prediction_mode(self, mode: Literal["deterministic", "bayesian"], n_samples: Optional[int] = None) -> None:
        if mode == "deterministic":
            self._pred_mode = mode
            if n_samples is not None:
                verification.check_equals(n_samples, 1)
        elif mode == "bayesian":
            self._pred_mode = mode
            verification.check_not_none(n_samples)
            if n_samples is not None:
                verification.check_le(1, n_samples)
        else:
            raise ValueError(f"Unknown prediction mode {mode}")

        if n_samples is not None:
            self._n_samples = n_samples

    def forward(self, x):
        # print(f"\n Forwarding with prediction mode {self._pred_mode} and prediction type {self._pred_type} and n_samples {self._n_samples}")
        original_device = next(self.dropout_hook.model.parameters()).device
        self.dropout_hook.model.to(self._device)
        if self._pred_mode == "deterministic":
            original_state = self.dropout_hook.enable_dropout
            self.dropout_hook.disable()
            logits = self.dropout_hook.model(x)
            probs  = nn.functional.softmax(logits, dim=-1)
            self.dropout_hook.enable_or_disable(original_state)
        elif self._pred_mode == "bayesian":
            self.dropout_hook.enable()
            logits = torch.stack([self.dropout_hook.model(x) for _ in range(self._n_samples)])
            probs = nn.functional.softmax(logits, dim=-1)
            self.dropout_hook.disable()
        else:
            raise ValueError(f"Unknown prediction mode {self._pred_mode}")
        self.dropout_hook.model.to(original_device)

        return probs

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("LaplaceModule does not support training_step")
    
    
    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def _eval_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        for metric_name in self.val_metrics.keys():
            if self._pred_mode == "bayesian":
                if hasattr(self.val_metrics[metric_name], "is_ensemble_metric") and self.val_metrics[metric_name].is_ensemble_metric:
                    preds = outputs
                else:
                    preds = outputs.mean(dim=0)
            else:
                preds = outputs
            self.val_metrics[metric_name](preds, labels)

        return outputs

    def on_validation_epoch_end(self) -> None:
        for metric_name in self.val_metrics.keys():
            self.log(f'val/{metric_name}', self.val_metrics[metric_name], prog_bar=True)
        return super().on_validation_epoch_end()
    
    def on_test_epoch_end(self) -> None:
        for metric_name in self.val_metrics.keys():
            self.log(f'test/{metric_name}', self.val_metrics[metric_name], prog_bar=True)
        return super().on_test_epoch_end()
