from typing import Dict, Text, Any

from config.mode import ModelMode
from config.bayesian.laplace import fc as fc_config, lenet as lenet_config, resnet as resnet_config, swin as swin_config

from config.bayesian.laplace import lightning, data, eval

def get_default_laplace_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_laplace_params()
    elif model_mode == "lenet5" or model_mode == "cifar10_lenet5":
        return lenet_config.get_default_laplace_params()
    elif model_mode == 'resnet20' or model_mode ==  'resnet32' or model_mode == 'resnet44':
        return resnet_config.get_default_laplace_params()
    elif model_mode == 'swin_t':
        return swin_config.get_default_laplace_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
def get_default_laplace_name(model_mode: ModelMode) -> Text:
    if model_mode == "fc":
        return fc_config.get_default_laplace_name()
    elif model_mode == "lenet5" or model_mode == "cifar10_lenet5":
        return lenet_config.get_default_laplace_name()
    elif model_mode == 'resnet20' or model_mode ==  'resnet32' or model_mode == 'resnet44':
        return resnet_config.get_default_laplace_name(model_mode)
    elif model_mode == 'swin_t':
        return swin_config.get_default_laplace_name(model_mode)
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    
def get_default_laplace_prior_optimization_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_prior_optimization_params()
    elif model_mode == "lenet5" or model_mode == "cifar10_lenet5":
        return lenet_config.get_default_prior_optimization_params()
    elif model_mode == 'resnet20' or model_mode ==  'resnet32' or model_mode == 'resnet44':
        return resnet_config.get_default_prior_optimization_params()
    elif model_mode == 'swin_t':
        return swin_config.get_default_prior_optimization_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
