from typing import Dict, Text
import torchmetrics
import metrics

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
    deterministic_metrics = get_default_deterministic_val_metrics(num_classes)
    special_ensemble_metrics = {
        "entropy": metrics.ShannonEntropy(dist="categorical"),
        "std": metrics.StandardDev(top_k="all"),
        "std_1": metrics.StandardDev(top_k=1),
        "std_3": metrics.StandardDev(top_k=3),
    }
    return {**deterministic_metrics, **special_ensemble_metrics}