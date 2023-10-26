

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Text, Union
import logging
import lightning.pytorch as pl
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.utils import data as torch_data
import torch.distributions as dists
from netcal import metrics as netcal
import laplace
import laplace.utils.matrix

from util import assertion, checkpoint, verification, data as data_utils
from util import lightning as lightning_util, plot as plot_util
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils
import metrics
import config

class SaveLayerVarianceCallback(pl.Callback):
    def __init__(self, stage: Literal['train', 'val', 'test'] = 'val'):
        self.stage = stage
        self._sampling_index: Dict[nn.Module, int] = {}
        self._batch_index: Dict[nn.Module, int] = {}
        self._variances: Dict[nn.Module, metrics.VarianceEstimator] = {}
        self._module_names: Dict[nn.Module, str] = {}


    def module_variances(self) -> Dict[nn.Module, torch.Tensor]:
        return {module: metric.get_metric_value() for (module, metric) in self._variances.items()}
    
    def named_variances(self) -> Dict[str, torch.Tensor]:
        return {self._module_names[module]: metric.get_metric_value() for (module, metric) in self._variances.items()}

    def on_validation_start(self, trainer, pl_module):
        # TODO: Check this line does not have any impact and delete it
        print("Validation is starting")
        if self.stage == 'val':
            if isinstance(pl_module, lightning.mc_dropout.McDropoutModule):
                self.register_submodules(pl_module.dropout_hook.model)
                self.register_module(pl_module.dropout_hook.model, 'model')
            # TODO: think if I should also log the results of the model after softmax
            # self.register_hooks(pl_module.laplace.model)
            # pl_module.laplace.model.register_forward_hook(lambda module, input, output: self.module_hook_fn(module, input, torch.softmax(output, dim=-1)))
            # self._module_names[pl_module.laplace.model] = 'model'

    # TODO: Check this function does not have any impact and delete it
    def on_validation_end(self, trainer, pl_module):
        print("Validation is ending")

    def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.stage == 'val':
            self.globally_set_batch_index(batch_idx)
            self.globally_set_sampling_index(0)
        return super().on_validation_batch_start(trainer, pl_module, batch, batch_idx, dataloader_idx)

    def register_submodules(self, module: nn.Module):
        for (sub_module_name, sub_module) in module.named_modules():
            if not hasattr(sub_module, 'weight'):
                continue
            self.register_module(sub_module, sub_module_name)

    def register_module(self, module: nn.Module, module_name: Optional[Text] = None):
        self._variances[module] = metrics.VarianceEstimator()
        self._sampling_index[module] = 0
        self._batch_index[module] = 0
        self._module_names[module] = module_name if module_name is not None else module.__class__.__name__
        module.register_forward_hook(self.module_hook_fn)

    def module_hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        # if module not in self._variances:
        #     self._variances[module] = metrics.VarianceEstimator()
        # if module not in self._sampling_index:
        #     self._sampling_index[module] = 0
        # if module not in self._batch_index:
        #     self._batch_index[module] = 0
        self._variances[module].feed_probs(output, gt_labels=None, samples=None,
                                           sampling_index=self._sampling_index[module], 
                                           batch_index=self._batch_index[module])
        self.increase_sampling_index(module)

    def get_layer_variance(self, module: nn.Module):
        if module not in self._variances:
            return None
        return self._variances[module].get_metric_value()
    
    def set_batch_index(self, module: nn.Module, batch_index: int):
        self._batch_index[module] = batch_index
    
    def globally_set_batch_index(self, batch_index: int):
        for module in self._batch_index.keys():
            self.set_batch_index(module, batch_index)

    def increase_batch_index(self, module: nn.Module):
        self._batch_index[module] += 1

    def globally_increase_batch_index(self):
        for module in self._batch_index.keys():
            self.increase_batch_index(module)

    def set_sampling_index(self, module: nn.Module, sampling_index: int):
        self._sampling_index[module] = sampling_index

    def globally_set_sampling_index(self, sampling_index: int):
        for module in self._sampling_index.keys():
            self.set_sampling_index(module, sampling_index)
    
    def increase_sampling_index(self, module: nn.Module):
        self._sampling_index[module] += 1

    def globally_increase_sampling_index(self):
        for module in self._sampling_index.keys():
            self.increase_sampling_index(module)
    
class LayerVarianceHook:
    def __init__(self):
        self._sampling_index: Dict[nn.Module, int] = {}
        self._batch_index: Dict[nn.Module, int] = {}
        self._variances: Dict[nn.Module, metrics.VarianceEstimator] = {}

    def module_hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if module not in self._variances:
            self._variances[module] = metrics.VarianceEstimator()
        if module not in self._sampling_index:
            self._sampling_index[module] = 0
        if module not in self._batch_index:
            self._batch_index[module] = 0
        self._variances[module].feed_probs(output, gt_labels=None, samples=None,
                                           sampling_index=self._sampling_index[module], 
                                           batch_index=self._batch_index[module])
        self._sampling_index[module] += 1

    def get_module_std_dev(self, module: nn.Module):
        if module not in self._variances:
            return None
        return self._variances[module].get_metric_value()
    
    def increase_batch_index(self, module: nn.Module):
        self._batch_index[module] += 1

    def generally_increase_batch_index(self):
        for module in self._batch_index.keys():
            self.increase_batch_index(module)

