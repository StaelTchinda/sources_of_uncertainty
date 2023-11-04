from typing import Dict, Text, Any
from torchvision import datasets, transforms
import pytorch_lightning as pl

from config import path as path_config
from config.data.utils import DatasetMode
from data.lightning import ambiguous_mnist



def get_default_dataset_params(dataset_mode: DatasetMode):
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    root=str(path_config.DATA_PATH)
    if dataset_mode == "train":
        return {
            "root": root,
            "transform": mnist_transform,
            "normalize": False
        }
    elif dataset_mode == "val":
        return {
            "root": root,
            "transform": mnist_transform,
            "normalize": False
        }
    elif dataset_mode == "test":
        return {
            "root": root,
            "transform": mnist_transform,
            "normalize": False
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_train_dataloader_params(dataset_mode: DatasetMode):
    batch_size: int = 32
    big_batch_size: int = 64

    # See https://blackhc.github.io/ddu_dirty_mnist/
    num_workers: int = 0
    if dataset_mode == "train":
        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": True,
            "pin_memory": False,
        }
    elif dataset_mode == "val":
        return {
            "batch_size": big_batch_size,
            "num_workers": num_workers,
            "shuffle": False,
            "pin_memory": False,
        }
    elif dataset_mode == "test":
        return {
            "batch_size": big_batch_size,
            "num_workers": num_workers,
            "shuffle": False,
            "pin_memory": False,
        }
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")

def get_default_datamodule_params():
    return {
        "train_ratio": 50000,
        "val_ratio": 10000,
        "train_dataset_params": get_default_dataset_params("train"),
        "val_dataset_params": get_default_dataset_params("val"),
        "test_dataset_params": get_default_dataset_params("test"),
        "train_dataloader_params": get_default_train_dataloader_params("train"),
        "val_dataloader_params": get_default_train_dataloader_params("val"),
        "test_dataloader_params": get_default_train_dataloader_params("test"),
    }

def get_default_datamodule() -> pl.LightningDataModule:
    params = get_default_datamodule_params()
    return ambiguous_mnist.AmbiguousMnistDataModule(
        **params
    )