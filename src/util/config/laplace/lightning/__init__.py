from typing import Optional, Dict, Text, Any
import laplace
import pytorch_lightning as pl
from torch import nn

from util.config.mode import ModelMode
from network.lightning import laplace as bayesian_laplace
import util.config.laplace.lightning.fc as fc_config
import util.config.laplace.lightning.lenet as lenet_config

def get_default_lightning_laplace_module(model_mode: ModelMode, laplace: laplace.ParametricLaplace) -> bayesian_laplace.LaplaceModule:
    if model_mode == "fc":
        return fc_config.get_default_lightning_laplace_module(laplace)
    elif model_mode == "lenet5":
        return lenet_config.get_default_lightning_laplace_module(laplace)
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")


def get_default_lightning_laplace_trainer(model_mode: ModelMode, additional_params: Optional[Dict[Text, Any]] = None) -> pl.Trainer:
    if model_mode == "fc":
        params = fc_config.get_default_lightning_laplace_trainer_params()
    elif model_mode == "lenet5":
        params = lenet_config.get_default_lightning_laplace_trainer_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
    if additional_params is not None:
        params.update(additional_params)

    return pl.Trainer(**params)

