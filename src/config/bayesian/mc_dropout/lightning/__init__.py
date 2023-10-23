from typing import Optional, Dict, Text, Any
import laplace
import pytorch_lightning as pl
from torch import nn

from config.mode import ModelMode
from network.lightning import mc_dropout as bayesian_mc_dropout
from config.bayesian.laplace.lightning import fc as fc_config, lenet as lenet_config, vgg as vgg_config
from network.bayesian import mc_dropout 

def get_default_mc_dropout_lightning_module_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        params = fc_config.get_default_laplace_lightning_module_params()
    elif model_mode == "lenet5":
        params = lenet_config.get_default_laplace_lightning_module_params()
    elif model_mode == "vgg11" or model_mode == "vgg13" or model_mode == "vgg16":
        params = vgg_config.get_default_laplace_lightning_module_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
    del params["pred_type"]

    return params

def get_default_lightning_mc_dropout_module(model_mode: ModelMode, dropout_hook: mc_dropout.DropoutHook) -> bayesian_mc_dropout.McDropoutModule:
    params = get_default_mc_dropout_lightning_module_params(model_mode)
    return bayesian_mc_dropout.McDropoutModule(dropout_hook, **params)


