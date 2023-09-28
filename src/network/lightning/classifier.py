from typing import Any, Callable, Dict, List, Text, Type, Optional
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from util import verification


class LightningClassifier(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module, 
                 loss_function: Callable, 
                 optimizer: Type[optim.Optimizer], 
                 optimizer_params: Dict[Text, Any], 
                 lr_scheduler: Optional[Type[optim.lr_scheduler._LRScheduler]] = None,
                 lr_scheduler_params: Optional[Dict[Text, Any]] = None,
                 train_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None, 
                 val_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None,):
        super(LightningClassifier, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_type = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler_type = lr_scheduler
        if lr_scheduler is not None:
            verification.check_not_none(lr_scheduler_params)
        self.lr_scheduler_params = lr_scheduler_params
        self.train_metrics = nn.ModuleDict(train_metrics if train_metrics is not None else {})
        self.val_metrics = nn.ModuleDict(val_metrics if val_metrics is not None else {})
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x):
        logits = self.model(x)
        if torch.allclose(logits.sum(dim=1), torch.zeros(logits.shape[0], device=logits.device)):
            warnings.warn("Logits sum to zero as it is already a probability distribution. Do you really want to apply softmax?")
        probs  = nn.functional.softmax(logits, dim=1)
        return probs
    
    def training_step(self, batch, batch_idx):
        # Define the training step logic here
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_function(preds, labels)
        self.log('train/loss', loss, sync_dist=True, on_epoch=True, on_step=True)
        for (metric_name, metric) in self.train_metrics.items():
            metric(preds, labels)
        return loss
    

    def on_train_epoch_end(self):
        for (metric_name, metric) in self.train_metrics.items():
            if isinstance(metric, torchmetrics.Metric):
                self.log(f'train/{metric_name}', metric.compute())
                metric.reset()
            else:
                raise ValueError(f"Unknown metric type {type(metric)}")
        super().on_train_epoch_end()
        
    
    def validation_step(self, batch, batch_idx):
        # Define the validation step logic here
        inputs, labels = batch
        probs = self(inputs)
        loss = self.loss_function(probs, labels)
        # print(f"LightningModule validation_step probs: {probs}, \n preds: {probs.argmax(dim=1)}")
        self.log('val/loss', loss, sync_dist=True)
        for metric_name in self.val_metrics.keys():
            self.val_metrics[metric_name](probs, labels)
        return loss

    def on_validation_epoch_end(self) -> None:
        for metric_name in self.val_metrics.keys():
            self.log(f'val/{metric_name}', self.val_metrics[metric_name])
        super().on_validation_epoch_end()

    def configure_optimizers(self):
        # Define your optimizer configuration here
        optimizer = self.optimizer_type(self.model.parameters(), **self.optimizer_params)
        if self.lr_scheduler_type is not None:
            lr_scheduler = self.lr_scheduler_type(optimizer, **self.lr_scheduler_params)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val/loss"}
        else:
            return optimizer