from typing import Optional, Dict, Text, Any
import laplace
import pytorch_lightning as pl
from torch import nn

from config.mode import ModelMode
from network.lightning import laplace as bayesian_laplace
from config.laplace.lightning import fc as fc_config, lenet as lenet_config, vgg as vgg_config
from network.pruning import laplace as laplace_pruning

def get_default_laplace_lightning_module_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_laplace_lightning_module_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_laplace_lightning_module_params()
    elif model_mode == "vgg11" or model_mode == "vgg13" or model_mode == "vgg16":
        return vgg_config.get_default_laplace_lightning_module_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")

def get_default_lightning_laplace_module(model_mode: ModelMode, laplace: laplace.ParametricLaplace) -> bayesian_laplace.LaplaceModule:
    params = get_default_laplace_lightning_module_params(model_mode)
    return bayesian_laplace.LaplaceModule(laplace, **params)


def get_default_lightning_laplace_pruning_module(model_mode: ModelMode, laplace: laplace.ParametricLaplace) -> laplace_pruning.LaplacePruningModule:
    params = get_default_laplace_lightning_module_params(model_mode)
    return laplace_pruning.LaplacePruningModule(laplace, **params)

def get_default_lightning_laplace_trainer_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_lightning_laplace_trainer_params()
    elif model_mode == "lenet5":
        params = lenet_config.get_default_lightning_laplace_trainer_params()
    elif model_mode == "vgg11" or model_mode == "vgg13" or model_mode == "vgg16":
        params = vgg_config.get_default_lightning_laplace_trainer_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    return params


def get_default_lightning_laplace_trainer(model_mode: ModelMode, additional_params: Optional[Dict[Text, Any]] = None) -> pl.Trainer:
    params = get_default_lightning_laplace_trainer_params(model_mode)
    
    if additional_params is not None:
        params.update(additional_params)

    return pl.Trainer(**params)

