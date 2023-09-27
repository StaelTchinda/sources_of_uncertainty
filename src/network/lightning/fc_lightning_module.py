from typing import Any, Callable, Dict, List, Text, Type, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class FcLightningModule(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module, 
                 loss_function: Callable, 
                 optimizer: Type[optim.Optimizer], 
                 optimizer_params: Dict[Text, Any], 
                 train_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None, 
                 val_metrics: Optional[Dict[Text, torchmetrics.Metric]] = None,):
        super(FcLightningModule, self).__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_type = optimizer
        self.optimizer_params = optimizer_params
        self.train_metrics = nn.ModuleDict(train_metrics if train_metrics is not None else {})
        self.val_metrics = nn.ModuleDict(val_metrics if val_metrics is not None else {})
        self.save_hyperparameters(ignore=['model', 'loss_function'])
    
    def forward(self, x):
        logits = self.model(x)
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
        return optimizer