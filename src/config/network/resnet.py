from typing import Dict, Text, Any
from torch import nn
from config.mode import ModelMode

from network.architecture.resnet import ResNet20, ResNet32, ResNet44


def get_default_model_params(model_mode: ModelMode) -> Dict[Text, Any]:
    return {
        'num_classes': 10,
        'dropout_p': 0.1,
    }

def get_default_model(model_mode: ModelMode) -> nn.Module:
    params = get_default_model_params(model_mode)
    if model_mode == "resnet20":
        model = ResNet20(**params)
    elif model_mode == "resnet32":
        model = ResNet32(**params)
    elif model_mode == "resnet44":
        model = ResNet44(**params)
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    return model
    