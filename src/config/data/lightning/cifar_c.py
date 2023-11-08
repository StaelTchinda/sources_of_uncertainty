from typing import Dict, Text, Tuple, Union

import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms
import pytorch_lightning as pl
import sklearn
import sklearn.model_selection
from config.mode import JointDataMode, JointDataMode

from util import data as data_util
from config import path as path_config
from config.data.utils import DatasetMode

from data.lightning import cifar
from data.lightning import cifar_c

# Hyperparameters from https://github.com/akamaster/pytorch_resnet_cifar10
def get_default_dataset_params(data_mode: JointDataMode, dataset_mode: DatasetMode):
    cifar10_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        # transforms.RandomAdjustSharpness(0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    root=str(path_config.DATA_PATH)
    corruptions: Dict[JointDataMode, Text] = {
        # "cifar10": "gaussian_noise",
        "cifar10_c_fog": "fog",
    }
    corruption = corruptions[data_mode]
    if dataset_mode == "train":
        return {
            "root": root,
            "corruption": corruption,
            "transform": cifar10_train_transform
        }
    elif dataset_mode == "val":
        return {
            "root": root,
            "corruption": corruption,
            "transform": cifar10_test_transform
        }
    elif dataset_mode == "test":
        return {
            "root": root,
            "corruption": corruption,
            "transform": cifar10_test_transform
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_train_dataloader_params(dataset_mode: DatasetMode):
    batch_size: int = 16 # 128 # For network 16 # Fpr Laplace
    big_batch_size: int = 16 # 128 # For network # 16 # Fpr Laplace
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
    elif dataset_mode == "test":
        return {
            "batch_size": big_batch_size,
            "num_workers": num_workers,
            "shuffle": False,
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_datamodule_params(data_mode: JointDataMode):
    return {
        "train_ratio": 40000,
        "val_ratio": 10000,
        "train_dataset_params": get_default_dataset_params(data_mode, "train"),
        "val_dataset_params": get_default_dataset_params(data_mode, "val"),
        "test_dataset_params": get_default_dataset_params(data_mode, "test"),
        "train_dataloader_params": get_default_train_dataloader_params("train"),
        "val_dataloader_params": get_default_train_dataloader_params("val"),
        "test_dataloader_params": get_default_train_dataloader_params("test"),
    }

def get_default_datamodule(data_mode: JointDataMode) -> pl.LightningDataModule:
    params = get_default_datamodule_params(data_mode)
    return cifar_c.Cifar10CDataModule(
        **params
    )