


from typing import Any, Dict, Text, Tuple

from torch.utils import data as torch_data

from util.config.mode import DataMode

from util.config.dataset import iris as iris_config, mnist as mnist_config, cifar as cifar_config, lightning


def get_default_datasets_params(data_mode: DataMode) -> Dict[Text, Any]:
    if data_mode == "iris":
        return iris_config.get_default_datasets_params()
    elif data_mode == "mnist":
        return mnist_config.get_default_datasets_params()
    elif data_mode == "cifar10":
        return cifar_config.get_default_datasets_params()
    else:
        raise NotImplementedError(f"Data mode {data_mode} not implemented")
    
def get_default_dataloaders_params(data_mode: DataMode) -> Dict[Text, Any]:
    if data_mode == "iris":
        return iris_config.get_default_dataloaders_params()
    elif data_mode == "mnist":
        return mnist_config.get_default_dataloaders_params()
    elif data_mode == "cifar10":
        return cifar_config.get_default_dataloaders_params()
    else:
        raise NotImplementedError(f"Data mode {data_mode} not implemented")


def get_default_dataloaders(data_mode: DataMode) -> Tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader]:
    if data_mode == "iris":
        return iris_config.get_default_dataloaders()
    elif data_mode == "mnist":
        return mnist_config.get_default_dataloaders()
    elif data_mode == "cifar10":
        return cifar_config.get_default_dataloaders()
    else:
        raise NotImplementedError(f"Data mode {data_mode} not implemented")
    

def get_default_laplace_dataloaders(data_mode: DataMode) -> Tuple[torch_data.DataLoader, torch_data.DataLoader, torch_data.DataLoader]:
    if data_mode == "iris":
        return iris_config.get_default_laplace_dataloaders()
    elif data_mode == "mnist":
        return mnist_config.get_default_laplace_dataloaders()
    else:
        raise NotImplementedError(f"Data mode {data_mode} not implemented")