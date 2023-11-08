from typing import Dict, Text, Any
from torch import nn

from config.mode import ModelMode
from config.network import fc as fc_config, lenet as lenet_config, resnet as resnet_config, swin as swin_config
from config.network.wideresnet import wideresnet50 as wideresnet50_config 

from config.network import lightning, checkpoint

def get_default_model_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "fc":
        return fc_config.get_default_model_params()
    elif model_mode == "lenet5":
        return lenet_config.get_default_model_params()
    elif model_mode == "resnet20" or model_mode == "resnet32" or model_mode == "resnet44":
        return resnet_config.get_default_model_params(model_mode)
    elif model_mode == 'swin_t':
        return swin_config.get_default_model_params(model_mode)
    elif model_mode == "wideresnet50":
        return wideresnet50_config.get_default_model_params()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")

def get_default_model(model_mode: ModelMode) -> nn.Module:
    if model_mode == "fc":
        return fc_config.get_default_model()
    elif model_mode == "lenet5":
        return lenet_config.get_default_model()
    elif model_mode == "resnet20" or model_mode == "resnet32" or model_mode == "resnet44":
        return resnet_config.get_default_model(model_mode)
    elif model_mode == 'swin_t':
        return swin_config.get_default_model(model_mode)
    elif model_mode == "wideresnet50":
        return wideresnet50_config.get_default_model()
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")