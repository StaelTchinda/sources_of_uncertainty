from typing import Text, Union, List, Optional, Mapping
from torch.utils import data as torch_data


from util import verification



def split_data(data: torch_data.Dataset, split_ratios: Union[List[float], List[int]]) -> List[torch_data.Dataset]:
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
        data, split_counts
    )
    return datasets


def count_labels(dataset: torch_data.Dataset) -> Mapping[int, int]:
    label_counts = {}
    for bash in dataset:
        X, y = bash[0], bash[1]
        label = int(y)
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    return label_counts


def verbose_dataloaders(train_dataloader: Optional[torch_data.DataLoader], val_dataloader: torch_data.DataLoader, test_dataloader: torch_data.DataLoader, verbose_class_distr: bool = False) -> Text:

    def print_count_labels(dataset):
        return ", ".join([f"{label}: {label_count}" for (label, label_count) in count_labels(dataset).items()])
    result_str = ""

    if train_dataloader is not None:
        result_str += f"Training set size: {len(train_dataloader.dataset)}\n"
        if verbose_class_distr:
            result_str += f"{print_count_labels(train_dataloader.dataset)}\n"
    result_str += f"Validation set size: {len(val_dataloader.dataset)}\n"
    if verbose_class_distr:
        result_str += f"{print_count_labels(val_dataloader.dataset)}\n"
    result_str += f"Test set size: {len(test_dataloader.dataset)}\n"
    if verbose_class_distr:
        result_str += f"{print_count_labels(test_dataloader.dataset)}\n"

    dataloader_to_sample = train_dataloader if train_dataloader is not None else val_dataloader
    for batch in dataloader_to_sample:
        X, y = batch[0], batch[1]
        result_str += f"Shape of X [B, C, H, W]: {X.shape}  {X.dtype}\n"
        result_str += f"Shape of y: {y.shape} {y.dtype}\n"
        break

    return result_str
