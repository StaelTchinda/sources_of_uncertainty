from typing import Optional, Tuple, List, Dict, Text
import torch

from callbacks.keep_sample import KeepImagesCallbackContainer, BatchFilter
from util import assertion


def mnist_keep_callback_filters(correct: bool):
    filters = []
    for label in range(10):
        filters.append(BatchFilter(correct=correct, label=label))
    return filters

def mnist_keep_all_callback_filters():
    filters = []
    for label in range(10):
        filters.append(BatchFilter(label=label))
    return filters

def mnist_keep_callback_scorers():
    def compute_variance(batch: Tuple[torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
        return outputs.var(1).mean(-1)

    return [
        compute_variance for _ in range(10)
    ]


def get_mnist_callback_containers() -> Dict[Text, KeepImagesCallbackContainer]:
    return {
        "correct_most_certain": KeepImagesCallbackContainer(mnist_keep_callback_filters(True), mnist_keep_callback_scorers(), highest=False),
        "incorrect_most_uncertain": KeepImagesCallbackContainer(mnist_keep_callback_filters(False), mnist_keep_callback_scorers(), highest=True),
        "correct_most_uncertain": KeepImagesCallbackContainer(mnist_keep_callback_filters(True), mnist_keep_callback_scorers(), highest=True),
        "incorrect_most_certain": KeepImagesCallbackContainer(mnist_keep_callback_filters(False), mnist_keep_callback_scorers(), highest=False),
        "most_certain": KeepImagesCallbackContainer(mnist_keep_all_callback_filters(), mnist_keep_callback_scorers(), highest=False),
        "most_uncertain": KeepImagesCallbackContainer(mnist_keep_all_callback_filters(), mnist_keep_callback_scorers(), highest=True)
    }
