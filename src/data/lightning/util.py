from typing import Any, Dict, List, Text, Sequence, Union, Optional, TypeVar
import warnings
import math

import torch
from torch.utils import data as torch_data
from torchvision import datasets
from data.lightning.datamodule import CustomizableDataModule


def adapt_dataset_size(dataset: torch_data.Dataset, ratios: List[int]) -> torch_data.Dataset:
    if sum(ratios) < len(dataset): # type: ignore[arg-type]
        return torch_data.Subset(dataset, range(sum(ratios)))
    else:
        return dataset
    
# Helper function inspired by torch.utils.data.random_split
T = TypeVar('T')
def random_indices_split(dataset: torch_data.Dataset[T], lengths: Sequence[Union[int, float]],
                         generator: Optional[torch.Generator] = torch.default_generator) -> List[List[int]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError(f"Sum of input lengths {sum(lengths)} does not equal the length {len(dataset)} of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [indices[offset - length : offset] for offset, length in zip(torch._utils._accumulate(lengths), lengths)]


