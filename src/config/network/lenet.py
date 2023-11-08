from typing import Dict, Text, Any
from torch import nn
from config.mode import ModelMode

from network.architecture.lenet import LeNet5


def get_default_model_params(model_mode: ModelMode) -> Dict[Text, Any]:
    if model_mode == "lenet5":
        return {
            'in_channels': 1,
            'num_classes': 10,
            'dropout_p': 0.2,
        }
    elif model_mode == "cifar10_lenet5":
        return {
            'in_channels': 3,
            'num_classes': 10,
            'dropout_p': 0.2,
        }
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    

def get_default_model(model_mode: ModelMode) -> nn.Module:
    params = get_default_model_params(model_mode)
    return LeNet5(**params)