from typing import Optional, Tuple, List, Dict, Text
import torch

from callbacks.keep_sample import KeepImagesCallbackContainer, BatchFilter
from util import assertion


# TODO: get rid of print statements

# TODO: check if not used and delete if not
def filter_predictions(batch: Tuple[torch.Tensor], outputs: torch.Tensor, correct: Optional[bool] = None, label: Optional[int] = None) -> List[int]:
    print(f"filter_predictions - correct: {correct}, label: {label}")
    if len(outputs.shape) == 2:
        probs = outputs
    elif len(outputs.shape) == 3:
        probs = outputs.mean(dim=1)
    else:
        raise ValueError(f"Output shape {outputs.shape} not supported")

    gt_labels = batch[1]
    pred_labels = probs.argmax(-1)

    mask = torch.ones_like(gt_labels, dtype=torch.bool)

    if correct is not None:
        if correct:
            mask = torch.logical_and(mask, gt_labels==pred_labels)
        else:
            mask = torch.logical_and(mask, gt_labels!=pred_labels)
    if label is not None:
        mask = torch.logical_and(mask, gt_labels==label)

    idxs = mask.nonzero().squeeze(-1).tolist()

    assertion.assert_le(gt_labels[idxs].unique().numel(), 1, f"Expected 1 classes, but got {gt_labels[idxs].unique().numel()}: {gt_labels[idxs].unique()}")
    if gt_labels[idxs].unique().numel() > 0:
        print(f"label: {label}, gt_labels[idxs].unique(): {gt_labels[idxs].unique()}")
        assertion.assert_equals(label, gt_labels[idxs].unique())
    if correct:
        assertion.assert_tensor_close(gt_labels[idxs], pred_labels[idxs])
    return idxs


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
        "most_certain": KeepImagesCallbackContainer(mnist_keep_all_callback_filters(), mnist_keep_callback_scorers(), highest=False),
        "most_uncertain": KeepImagesCallbackContainer(mnist_keep_all_callback_filters(), mnist_keep_callback_scorers(), highest=True)
    }
