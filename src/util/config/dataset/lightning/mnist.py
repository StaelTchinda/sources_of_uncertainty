from typing import Dict, Text, Any
from torchvision import datasets, transforms
import pytorch_lightning as pl

from util.config import path as path_config
from data.dataset.lightning import mnist

def get_default_datamodule_parms() -> Dict[Text, Any]:
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return {
        "batch_size": 32,
        "data_dir": str(path_config.DATA_PATH),
        "transform": transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        "train_ratio": 50000,
        "val_ratio": 10000
    }

def get_default_datamodule() -> pl.LightningDataModule:
    params = get_default_datamodule_parms()
    return mnist.MNISTDataModule(
        **params
    )