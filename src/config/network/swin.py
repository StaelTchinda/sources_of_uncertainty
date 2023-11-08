from typing import Dict, Text, Any
from torch import nn
import torchvision

from config.mode import ModelMode
from network.architecture.resnet import ResNet20, ResNet32, ResNet44
from torchvision.models.swin_transformer import Swin_T_Weights

def get_default_model_params(model_mode: ModelMode) -> Dict[Text, Any]:
    return {
        'num_classes': 10,
        'dropout_p': 0.1,
    }

def get_default_model(model_mode: ModelMode) -> nn.Module:
    params = get_default_model_params(model_mode)
    if model_mode == "swin_t":
        # TODO: update the dropout
        model = torchvision.models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1, progress=True)
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")
    return model
    