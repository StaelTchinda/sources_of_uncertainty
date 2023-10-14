from typing import Literal, Optional, Dict, Text

from laplace import ParametricLaplace

import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics

import laplace as laplace_lib
from laplace import utils as laplace_libutils
import laplace.utils.matrix

from util import verification, assertion, device as device_util

PredictionMode = Literal["deterministic", "bayesian"]

class LaplaceModule(pl.LightningModule):
    def __init__(self, laplace: ParametricLaplace, prediction_mode: PredictionMode = "deterministic", pred_type: Literal["glm", "nn"]="nn", n_samples: int = 1, val_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None):
        super(LaplaceModule, self).__init__()
        verification.check_not_none(laplace)
        verification.check_not_none(laplace.model)
        self.laplace = laplace
        device_util.move_laplace_to_device(self.laplace, self._device)
        self.laplace.model.to(self._device)
        self.configure_prediction_mode(prediction_mode, pred_type, n_samples)
        self.val_metrics = nn.ModuleDict(val_metrics if val_metrics is not None else {})
        self.save_hyperparameters()


    def configure_prediction_mode(self, mode: Literal["deterministic", "bayesian"], pred_type: Optional[Literal["glm", "nn"]]=None, n_samples: Optional[int] = None) -> None:
        if mode == "deterministic":
            self._pred_mode = mode
            if n_samples is not None:
                verification.check_equals(n_samples, 1)
        elif mode == "bayesian":
            self._pred_mode = mode
            verification.check_not_none(pred_type)
            verification.check_not_none(n_samples)
            if n_samples is not None:
                verification.check_le(1, n_samples)
        else:
            raise ValueError(f"Unknown prediction mode {mode}")

        self._pred_type = pred_type
        if n_samples is not None:
            self._n_samples = n_samples

    def forward(self, x):
        # print(f"\n Forwarding with prediction mode {self._pred_mode} and prediction type {self._pred_type} and n_samples {self._n_samples}")
        if self._pred_mode == "deterministic":
            original_device = device_util.move_model_to_device(self.laplace.model, self._device)
            # assertion.assert_tensor_close(self.laplace.mean.to(self.device), torch.nn.utils.convert_parameters.parameters_to_vector(self.laplace.model.parameters()).to(self.device))
            logits = self.laplace.model(x)
            device_util.move_model_to_device(self.laplace.model, original_device)
            probs  = nn.functional.softmax(logits, dim=-1)
            return probs
        elif self._pred_mode == "bayesian":
            original_device = device_util.move_laplace_to_device(self.laplace, self._device)
            probs = self.laplace.predictive_samples(x, pred_type=self._pred_type, n_samples=self._n_samples)
            device_util.move_laplace_to_device(self.laplace, original_device)
            return probs
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("LaplaceModule does not support training_step")
    
    
    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
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

    def test_step(self, batch, batch_idx):
        # Define the validation step logic here
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
            self.log(f'val/{metric_name}', self.val_metrics[metric_name])
        return super().on_validation_epoch_end()