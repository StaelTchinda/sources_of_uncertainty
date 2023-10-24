from typing import Optional, Tuple, List, Dict, Text
import torch

from callbacks.keep_sample import KeepImagesCallbackContainer, BatchFilter
from util import assertion
from config.bayesian.laplace.eval.util import keep_callback_filters, keep_all_callback_filters, compute_accuracy


def cifar10_keep_callback_scorers():
    return [
        compute_accuracy for _ in range(10)
    ]


def get_cifar10_callback_containers() -> Dict[Text, KeepImagesCallbackContainer]:
    return {
        "correct": KeepImagesCallbackContainer(keep_callback_filters(True), cifar10_keep_callback_scorers(), highest=True),
        "incorrect": KeepImagesCallbackContainer(keep_callback_filters(False), cifar10_keep_callback_scorers(), highest=True),
    }
