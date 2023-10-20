from typing import Dict, Text, Any
import torch
from torch import nn
import torchmetrics
from torch.utils import data as torch_data

from network.lightning.laplace import LaplaceModule

from metrics.shannon_entropy import ShannonEntropy
from metrics.standard_dev import StandardDev, standard_dev
from metrics.variance import VarianceEstimator
from metrics.mutual_information import MutualInformation
from metrics.variation_ratio import VariationRatio

def evaluate_model(model: nn.Module, dataloader: torch_data.DataLoader, metrics: Dict[Text, torchmetrics.Metric], device: torch.device):
    model.to(device)
    for batch in dataloader:
        X, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(X)
        for (metric_name, metric) in metrics.items():
            metric.to(device)
            metric(y_pred, y)


def evaluate_laplace_module(laplace_module: LaplaceModule, dataloader: torch_data.DataLoader, metrics: Dict[Text, torchmetrics.Metric], device: torch.device):
    laplace_module.to(device)
    for batch in dataloader:
        X, y = batch[0].to(device), batch[1].to(device)
        outputs = laplace_module(X)
        for (metric_name, metric) in metrics.items():
            if laplace_module._pred_mode == "bayesian":
                if hasattr(metric, "is_ensemble_metric") and metric.is_ensemble_metric:
                    y_pred = outputs
                else:
                    y_pred = outputs.mean(dim=0).to(device=outputs.device)
            else:
                y_pred = outputs
            metric.to(device)
            metric(y_pred, y)

    return metrics