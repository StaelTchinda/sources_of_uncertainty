from typing import Literal, Optional, Dict, Text

from laplace import ParametricLaplace

import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics

import laplace as laplace_lib
from laplace import utils as laplace_libutils
import laplace.utils.matrix

from util import verification, assertion

PredictionMode = Literal["deterministic", "bayesian"]

class LaplaceModule(pl.LightningModule):
    def __init__(self, laplace: ParametricLaplace, prediction_mode: PredictionMode = "deterministic", pred_type: Literal["glm", "nn"]="nn", n_samples: int = 1, val_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None):
        super(LaplaceModule, self).__init__()
        verification.check_not_none(laplace)
        verification.check_not_none(laplace.model)
        self.laplace = laplace
        move_laplace_to_device(self.laplace, self._device)
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
            original_device = move_model_to_device(self.laplace.model, self._device)
            assertion.assert_tensor_close(self.laplace.mean.to(self.device), torch.nn.utils.convert_parameters.parameters_to_vector(self.laplace.model.parameters()).to(self.device))
            logits = self.laplace.model(x)
            move_model_to_device(self.laplace.model, original_device)
            probs  = nn.functional.softmax(logits, dim=-1)
            return probs
        elif self._pred_mode == "bayesian":
            original_device = move_laplace_to_device(self.laplace, self._device)
            probs = self.laplace.predictive_samples(x, pred_type=self._pred_type, n_samples=self._n_samples)
            move_laplace_to_device(self.laplace, original_device)
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
                    preds = outputs.mean(dim=0).to(device=outputs.device)
            else:
                preds = outputs
            self.val_metrics[metric_name](preds, labels)

        return outputs


    def on_validation_epoch_end(self) -> None:
        for metric_name in self.val_metrics.keys():
            self.log(f'val/{metric_name}', self.val_metrics[metric_name])
        return super().on_validation_epoch_end()

def move_model_to_device(model: nn.Module, device: torch.device) -> torch.device:
    original_device = next(model.parameters()).device
    model.to(device)
    return original_device

def move_laplace_to_device(laplace: laplace_lib.ParametricLaplace, device: torch.device) -> torch.device:
    original_device = laplace._device
    laplace._device = device
    laplace.model.to(device)
    laplace.prior_precision = laplace.prior_precision.to(device)
    if isinstance(laplace, laplace_lib.FullLaplace):
        laplace.H = laplace.H.to(device)
    elif isinstance(laplace, laplace_lib.KronLaplace):
        move_kron_decomposed_to_device(laplace.H, device)
    laplace.mean = laplace.mean.to(device)
    laplace._sigma_noise = laplace._H_factor.to(device)
    return original_device

def move_kron_decomposed_to_device(kron_decomposed: laplace_libutils.KronDecomposed, device: torch.device) -> torch.device:
    original_device = kron_decomposed.eigenvalues[0][0].device
    for (eigvec_idx) in range(len(kron_decomposed.eigenvectors)):
        for (block_idx) in range(len(kron_decomposed.eigenvectors[eigvec_idx])):
            kron_decomposed.eigenvectors[eigvec_idx][block_idx] = kron_decomposed.eigenvectors[eigvec_idx][block_idx].to(device)
    for (eigval_idx) in range(len(kron_decomposed.eigenvalues)):
        for (block_idx) in range(len(kron_decomposed.eigenvalues[eigval_idx])):
            kron_decomposed.eigenvalues[eigval_idx][block_idx] = kron_decomposed.eigenvalues[eigval_idx][block_idx].to(device)
    kron_decomposed.deltas = kron_decomposed.deltas.to(device)
    return original_device