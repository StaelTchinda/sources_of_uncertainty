from typing import Dict, Text, Any

from config.mode import ModelMode
from config.laplace import fc as fc_config, lenet as lenet_config, vgg as vgg_config

from config.laplace import lightning as lightning
from config.laplace import data

def get_default_laplace_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_laplace_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_laplace_params()
    elif model_mode == "vgg11" or model_mode == "vgg13" or model_mode == "vgg16":
        return vgg_config.get_default_laplace_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
def get_default_laplace_name(model_mode: ModelMode) -> Text:
    if model_mode == "fc":
        return fc_config.get_default_laplace_name()
    elif model_mode == "lenet5":
        return lenet_config.get_default_laplace_name()
    elif model_mode == "vgg11" or model_mode == "vgg13" or model_mode == "vgg16":
        return vgg_config.get_default_laplace_name()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
def get_default_laplace_prior_optimization_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_prior_optimization_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_prior_optimization_params()
    elif model_mode == "vgg11" or model_mode == "vgg13" or model_mode == "vgg16":
        return vgg_config.get_default_prior_optimization_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
