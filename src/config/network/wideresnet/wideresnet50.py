from typing import Dict, Text, Any
from torch import nn

from network.architecture.wideresnet import WideResNet50


def get_default_model_params() -> Dict[Text, Any]:
    return {
        # 'weights': VGG11_Weights.IMAGENET1K_V1,
        'num_classes': 182,
        'dropout_p': 0.5,
    }

def get_default_model() -> nn.Module:
    params = get_default_model_params()
    return WideResNet50(**params)