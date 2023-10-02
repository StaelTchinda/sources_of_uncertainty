from typing import Dict, Text, Any
from torch import nn

from network.architecture.vgg import VGG11


def get_default_model_params() -> Dict[Text, Any]:
    return {
        # 'weights': VGG11_Weights.IMAGENET1K_V1,
        'num_classes': 10,
        'dropout_p': 0.5,
    }

def get_default_model() -> nn.Module:
    params = get_default_model_params()
    return VGG11(**params)