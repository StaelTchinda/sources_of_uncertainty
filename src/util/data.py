from typing import List, Tuple, Union
import torch
from torch.utils import data as torch_data


from data.dataset.uci import UCIDataset
from util import verification


def split_data(data: torch_data.Dataset, split_ratios: Union[List[float], List[int]], generator: torch.Generator = None) -> List[torch_data.Dataset]:
    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(0)

    dataset_size: int = len(data)
    verification.check_not_equals(0, len(split_ratios))
    split_counts: List[int]
    if isinstance(split_ratios[0], int):
        verification.check_equals(dataset_size, sum(split_ratios))
        split_counts = split_ratios
    else:
        verification.check_float_close(1.0, sum(split_ratios))
        split_counts = [int(ratio * dataset_size) for ratio in split_ratios]

    if sum(split_counts) < dataset_size:
        i = 0
        while sum(split_counts) < dataset_size:
            split_counts[i] += 1
            i = i + 1 if i < dataset_size - 1 else 0
    datasets = torch_data.random_split(
        data, split_counts, generator=generator
    )
    return datasets