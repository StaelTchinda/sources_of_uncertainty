from typing import Dict, Text, Any
import pytorch_lightning as pl

from config import path as path_config
from config.data.utils import DatasetMode
from data.lightning import uci



def get_default_dataset_params(dataset_mode: DatasetMode):
    return {
        "name": "iris",
        "root": str(path_config.DATA_PATH)
    }


def get_default_dataloader_params(dataset_mode: DatasetMode):
    batch_size: int = 20
    num_workers: int = 4
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }


def get_default_datamodule_params():
    return {
        "train_ratio": 109,
        "val_ratio": 20,
        "test_ratio": 20,
        "train_dataset_params": get_default_dataset_params("train"),
        "val_dataset_params": get_default_dataset_params("val"),
        "test_dataset_params": get_default_dataset_params("test"),
        "train_dataloader_params": get_default_dataloader_params("train"),
        "val_dataloader_params": get_default_dataloader_params("val"),
        "test_dataloader_params": get_default_dataloader_params("test"),
    }

def get_default_datamodule() -> pl.LightningDataModule:
    params = get_default_datamodule_params()
    return uci.UciDataModule(
        **params
    )