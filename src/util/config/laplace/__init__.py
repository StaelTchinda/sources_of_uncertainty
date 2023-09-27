from typing import Dict, Text, Any

from util.config.mode import ModelMode
from util.config.laplace import fc as fc_config, lenet as lenet_config

from util.config.laplace import lightning as lightning

def get_default_laplace_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_laplace_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_laplace_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
def get_default_laplace_name(model_mode: ModelMode) -> Text:
    if model_mode == "fc":
        return fc_config.get_default_laplace_name()
    elif model_mode == "lenet5":
        return lenet_config.get_default_laplace_name()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
def get_default_laplace_prior_optimization_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_prior_optimization_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_prior_optimization_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
