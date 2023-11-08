from typing import Optional, Dict, Text, Any
import laplace
import pytorch_lightning as pl
from torch import nn

from config.mode import ModelMode
from network.lightning import laplace as bayesian_laplace
from config.network.lightning import fc as fc_config, lenet as lenet_config, resnet as resnet_config, wideresnet as wideresnet_config, swin as swin_config

def get_default_lightning_module(model_mode: ModelMode, model: nn.Module) -> pl.LightningModule:
    if model_mode == "fc":
        return fc_config.get_default_lightning_module(model)
    elif model_mode == "lenet5":
        return lenet_config.get_default_lightning_module(model)
    elif model_mode == 'resnet20' or model_mode ==  'resnet32' or model_mode == 'resnet44':
        return resnet_config.get_default_lightning_module(model)
    elif model_mode == 'swin_t':
        return swin_config.get_default_lightning_module(model)
    elif model_mode == "wideresnet50":
        return wideresnet_config.get_default_lightning_module(model)
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")


def get_default_lightning_trainer_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_lightning_trainer_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_lightning_trainer_params()
    elif model_mode == 'resnet20' or model_mode ==  'resnet32' or model_mode == 'resnet44':
        return resnet_config.get_default_lightning_trainer_params()
    elif model_mode == 'swin_t':
        return swin_config.get_default_lightning_trainer_params()
    elif model_mode == "wideresnet50":
        return wideresnet_config.get_default_lightning_trainer_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")


def get_default_lightning_trainer(model_mode: ModelMode, additional_params: Optional[Dict[Text, Any]] = None) -> pl.Trainer:
    params = get_default_lightning_trainer_params(model_mode)
    if additional_params is not None:
        params.update(additional_params)

    return pl.Trainer(**params)

