from typing import Optional, Tuple, List, Dict, Text
import torch

from callbacks.keep_sample import KeepImagesCallbackContainer, BatchFilter
from util import assertion
from config.bayesian.laplace.eval.util import keep_callback_filters, keep_all_callback_filters, compute_variance


def mnist_keep_callback_scorers():
    return [
        compute_variance for _ in range(10)
    ]


def get_mnist_callback_containers() -> Dict[Text, KeepImagesCallbackContainer]:
    return {
        "correct_most_certain": KeepImagesCallbackContainer(keep_callback_filters(True), mnist_keep_callback_scorers(), highest=False),
        "incorrect_most_uncertain": KeepImagesCallbackContainer(keep_callback_filters(False), mnist_keep_callback_scorers(), highest=True),
        "correct_most_uncertain": KeepImagesCallbackContainer(keep_callback_filters(True), mnist_keep_callback_scorers(), highest=True),
        "incorrect_most_certain": KeepImagesCallbackContainer(keep_callback_filters(False), mnist_keep_callback_scorers(), highest=False),
        "most_certain": KeepImagesCallbackContainer(keep_all_callback_filters(), mnist_keep_callback_scorers(), highest=False),
        "most_uncertain": KeepImagesCallbackContainer(keep_all_callback_filters(), mnist_keep_callback_scorers(), highest=True)
    }
