from typing import Dict, Text, Any
from torch import nn

from util.config.mode import ModelMode
from util.config.network import fc as fc_config, lenet as lenet_config, vgg as vgg_config

from util.config.network import lightning

def get_default_model_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_model_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_model_params()
    elif model_mode == "vgg11":
        return vgg_config.get_default_model_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")

def get_default_model(model_mode: ModelMode) -> nn.Module:
    if model_mode == "fc":
        return fc_config.get_default_model()
    elif model_mode == "lenet5":
        return lenet_config.get_default_model()
    elif model_mode == "vgg11":
        return vgg_config.get_default_model()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")