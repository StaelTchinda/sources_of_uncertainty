

import pytorch_lightning as pl

from config.mode import DataMode
from config.data.lightning import iris as iris_config, mnist as mnist_config, cifar as cifar_config, wildcam as wildcam_config

def get_default_datamodule(data_mode: DataMode) -> pl.LightningDataModule:
    if data_mode == "iris":
        return iris_config.get_default_datamodule()
    elif data_mode == "mnist":
        return mnist_config.get_default_datamodule()
    elif data_mode == "cifar10":
        return cifar_config.get_default_datamodule()
    elif data_mode == "wildcam":
        return wildcam_config.get_default_datamodule()
    else:
        raise ValueError(f"Unknown data mode {data_mode}")
