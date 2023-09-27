

import pytorch_lightning as pl

from util.config.mode import DataMode
from util.config.dataset.lightning import mnist as mnist_config, cifar as cifar_config

def get_default_datamodule(data_mode: DataMode) -> pl.LightningDataModule:
    if data_mode == "mnist":
        return mnist_config.get_default_datamodule()
    elif data_mode == "cifar10":
        return cifar_config.get_default_datamodule()
    else:
        raise ValueError(f"Unknown data mode {data_mode}")
