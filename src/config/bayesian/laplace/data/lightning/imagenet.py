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

from data.lightning import imagenet
from config.data.lightning import imagenet as imagenet_config

def get_default_dataset_params(dataset_mode: DatasetMode):
    return imagenet_config.get_default_dataset_params(dataset_mode)

def get_default_dataloader_params(dataset_mode: DatasetMode):
    batch_size: int = 16 #64
    num_workers: int = 4 # 4

    if dataset_mode == "train":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": True,
        }
    elif dataset_mode == "val":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        }
    elif dataset_mode == "test":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_datamodule_params():
    return {
        **imagenet_config.get_default_datamodule_params(),
        "train_dataloader_params": get_default_dataloader_params("train"),
        "val_dataloader_params": get_default_dataloader_params("val"),
        "test_dataloader_params": get_default_dataloader_params("test"),
    }

def get_default_datamodule() -> pl.LightningDataModule:
    params = get_default_datamodule_params()
    return imagenet.ImageNetDataModule(
        **params
    )