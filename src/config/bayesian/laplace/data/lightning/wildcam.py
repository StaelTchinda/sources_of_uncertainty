from typing import Tuple, Union

import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms
import pytorch_lightning as pl
import sklearn
import sklearn.model_selection

from util import data as data_util
from config import path as path_config
from config.data.utils import DatasetMode

from data.lightning import wildcam

def get_default_dataset_params(dataset_mode: DatasetMode):
    cifar10_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    root=str(path_config.DATA_PATH)
    if dataset_mode == "train":
        return {
            "root": root,
            "transform": cifar10_train_transform
        }
    elif dataset_mode == "val":
        return {
            "root": root,
            "transform": cifar10_test_transform
        }
    elif dataset_mode == "test":
        return {
            "root": root,
            "transform": cifar10_test_transform
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")


def get_default_train_dataloader_params(dataset_mode: DatasetMode):
    batch_size: int = 32
    num_workers: int = 4
    if dataset_mode == "train":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
    elif dataset_mode == "val":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
    elif dataset_mode == "test":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_datamodule_params():
    return {
        "train_dataset_params": get_default_dataset_params("train"),
        "val_dataset_params": get_default_dataset_params("val"),
        "test_dataset_params": get_default_dataset_params("test"),
        "train_dataloader_params": get_default_train_dataloader_params("train"),
        "val_dataloader_params": get_default_train_dataloader_params("val"),
        "test_dataloader_params": get_default_train_dataloader_params("test"),
    }

def get_default_datamodule() -> pl.LightningDataModule:
    params = get_default_datamodule_params()
    return wildcam.WildcamDataModule(
        **params
    )