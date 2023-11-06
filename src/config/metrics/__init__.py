from typing import Dict, Text
import torchmetrics
import metrics
import torch
from torch import nn
import copy
from util import assertion

def get_default_deterministic_train_metrics(num_classes: int) -> Dict[Text, torchmetrics.Metric]:
    return {
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "ece": torchmetrics.CalibrationError(num_classes=num_classes, task="multiclass", n_bins=10, norm="l1"), # ECE
    }

def get_default_deterministic_val_metrics(num_classes: int) -> Dict[Text, torchmetrics.Metric]:
    return {
        "acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        "f1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        "auroc": torchmetrics.AUROC(num_classes=num_classes, average="macro", task="multiclass"),
        "ap": torchmetrics.AveragePrecision(num_classes=num_classes, average="macro", task="multiclass"),
        "ece": torchmetrics.CalibrationError(num_classes=num_classes, task="multiclass", n_bins=10, norm="l1"), # ECE
    }

def get_default_ensemble_val_metrics(num_classes: int) -> Dict[Text, torchmetrics.Metric]:
    deterministic_metrics: Dict[Text, torchmetrics.Metric] = get_default_deterministic_val_metrics(num_classes)
    stochastic_metrics: Dict[Text, torchmetrics.Metric]  = {
        "mutual_information": metrics.MutualInformation(),
        "entropy": metrics.ShannonEntropy(dist="categorical"),
        "variation_ratio": metrics.VariationRatio(num_classes=num_classes),
        "std": metrics.StandardDev(top_k="all"),
        "std_1": metrics.StandardDev(top_k=1),
        "std_3": metrics.StandardDev(top_k=3),
    }

    def filter_correct_preds(module: nn.Module, input) -> torch.Tensor:
        probs: torch.Tensor
        targets: torch.Tensor
        probs, targets = input

        assertion.assert_contains([2, 3], len(probs.shape), f"Expected probs to have shape (S, B, C) or (B, C), but got {probs.shape}")
        if len(probs.shape) == 3:
            mask = (probs.mean(dim=0).argmax(dim=-1) == targets)
            return probs[:, mask, :], targets[mask]
        else:
            mask = (probs.argmax(dim=-1) == targets)
            return probs[mask, :], targets[mask]
    
    def filter_incorrect_preds(module: nn.Module, input) -> torch.Tensor:
        probs: torch.Tensor
        targets: torch.Tensor
        probs, targets = input

        assertion.assert_contains([2, 3], len(probs.shape), f"Expected probs to have shape (S, B, C) or (B, C), but got {probs.shape}")
        if len(probs.shape) == 3:
            mask = (probs.mean(dim=0).argmax(dim=-1) == targets)
            return probs[:, ~mask, :], targets[~mask]
        else:
            mask = (probs.argmax(dim=-1) == targets)
            return probs[~mask, :], targets[~mask]
    
    only_correct_stochastic_metrics: Dict[Text, torchmetrics.Metric] = { 
        f"pos_{key}": copy.deepcopy(value) for key, value in stochastic_metrics.items()
    }
    for metric in only_correct_stochastic_metrics.values():
        metric.register_forward_pre_hook(filter_correct_preds)

    only_incorrect_stochastic_metrics: Dict[Text, torchmetrics.Metric] = {
        f"neg_{key}": copy.deepcopy(value) for key, value in stochastic_metrics.items()
    }
    for metric in only_incorrect_stochastic_metrics.values():
        metric.register_forward_pre_hook(filter_incorrect_preds)

    return {**deterministic_metrics, **stochastic_metrics, **only_correct_stochastic_metrics, **only_incorrect_stochastic_metrics}