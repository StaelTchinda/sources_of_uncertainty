
from typing import Dict, Literal, Optional, Text, Union
import pytorch_lightning as pl


import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.utils.prune as prune
import torchmetrics
from network.lightning import laplace as lightning_laplace
from network.pruning.pruner import PruningStrategy
from network.pruning.util import get_parameter_mask
from network.pruning.pruner import Pruner

from util import assertion, verification, device as device_util

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
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import laplace

# Inspired from laplace.baselaplace.ParametricLaplace._nn_predictive_samples
# TODO: delete function if unused
def laplace_pruned_nn_predictive_samples(self: laplace.ParametricLaplace, X: torch.Tensor, n_samples: int = 100):
    fs = list()
    parameter_mask = get_parameter_mask(self.model)
    for sample in self.sample(n_samples):
        sample = sample * parameter_mask
        vector_to_parameters(sample, self.model.parameters())
        f = self.model(X.to(self._device))
        fs.append(f.detach() if not self.enable_backprop else f)
    vector_to_parameters(self.mean, self.model.parameters())
    fs = torch.stack(fs)
    if self.likelihood == 'classification':
        fs = torch.softmax(fs, dim=-1)
    return fs

class LaplacePruningModule(lightning_laplace.LaplaceModule):

    def __init__(self, laplace: ParametricLaplace, pruner: Optional[Pruner] = None, **kwargs):
        super(LaplacePruningModule, self).__init__(laplace, **kwargs)
        if pruner is not None:
            self.pruner = pruner
            verification.check_is(self.pruner.original_model, self.laplace.model)
        else:
            self.pruner = Pruner(self.laplace.model)

    def forward(self, x):
        # print(f"\n Forwarding with prediction mode {self._pred_mode} and prediction type {self._pred_type} and n_samples {self._n_samples}")
        if self._pred_mode == "deterministic":
            original_device = device_util.move_model_to_device(self.laplace.model, self._device)
            assertion.assert_tensor_close(self.laplace.mean.to(self.device), torch.nn.utils.convert_parameters.parameters_to_vector(self.laplace.model.parameters()).to(self.device))
            logits = self.laplace.model(x)
            device_util.move_model_to_device(self.laplace.model, original_device)
            probs  = nn.functional.softmax(logits, dim=-1)
            return probs
        elif self._pred_mode == "bayesian":
            # verification.check_equals("nn", self._pred_type, f"Only nn is supported for now, but got {self._pred_type}")
            original_device = device_util.move_laplace_to_device(self.laplace, self._device)
            # probs = laplace_pruned_nn_predictive_samples(self.laplace, x, n_samples=self._n_samples)
            probs = self.laplace.predictive_samples(x, pred_type=self._pred_type, n_samples=self._n_samples)
            device_util.move_laplace_to_device(self.laplace, original_device)
            return probs

    def on_validation_epoch_end(self) -> None:
        self._on_eval_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._on_eval_epoch_end()

    def _on_eval_epoch_end(self):
        for (metric_name, metric) in self.val_metrics.items():
            metric_value = metric.compute()
            metric.reset()
            self.logger.experiment.add_scalar(f'prune/{metric_name}', metric_value, global_step=int(100*self.pruner.pruning_sparsity))
        return pl.LightningModule.on_validation_epoch_end(self)
