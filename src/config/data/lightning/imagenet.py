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

# Hyperparameters from https://github.com/akamaster/pytorch_resnet_imagenet
def get_default_dataset_params(dataset_mode: DatasetMode):
    imagenet_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    imagenet_val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    root=str(path_config.DATA_PATH / "ImageNet")
    if dataset_mode == "train":
        return {
            "root": root,
            "transform": imagenet_train_transform
        }
    elif dataset_mode == "val":
        return {
            "root": root,
            "transform": imagenet_val_transform
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_train_dataloader_params(dataset_mode: DatasetMode):
    batch_size: int = 2
    big_batch_size: int = 2
    num_workers: int = 4

    if dataset_mode == "train":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": True,
        }
    elif dataset_mode == "val":
        return {
            "batch_size": big_batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_datamodule_params():
    return {
        "train_ratio": None,
        "val_ratio": 10,
        "train_dataset_params": get_default_dataset_params("train"),
        "val_dataset_params": get_default_dataset_params("val"),
        "train_dataloader_params": get_default_train_dataloader_params("train"),
        "val_dataloader_params": get_default_train_dataloader_params("val"),
    }

def get_default_datamodule() -> pl.LightningDataModule:
    params = get_default_datamodule_params()
    return imagenet.ImageNetDataModule(
        **params
    )