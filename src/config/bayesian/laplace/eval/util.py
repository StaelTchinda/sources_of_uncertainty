from typing import Optional, Tuple, List, Dict, Text
import torch

from callbacks.keep_sample import KeepImagesCallbackContainer, BatchFilter
from util import assertion


def keep_callback_filters(correct: bool):
    filters = []
    for label in range(10):
        filters.append(BatchFilter(correct=correct, label=label))
    return filters

def keep_all_callback_filters():
    filters = []
    for label in range(10):
        filters.append(BatchFilter(label=label))
    return filters

def compute_variance(batch: Tuple[torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
    return outputs.var(1).mean(-1)

def compute_accuracy(batch: Tuple[torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
    gt_labels = batch[1]
    # If the probs are bayesian, then we need to take the mean of the probs
    if len(outputs.shape) == 3:
        outputs = outputs.mean(1)
    predicted = torch.argmax(outputs, dim=-1)
    correct = (predicted == gt_labels).float()
    accuracy = correct / len(gt_labels)
    return accuracy