from typing import Dict, Text, Any
from torch import nn

from network.architecture.lenet import LeNet5
from network.architecture.vgg import VGG16
from torchvision.models import VGG16_Weights, VGG16_BN_Weights, VGG, vgg16


def get_default_model_params() -> Dict[Text, Any]:
    return {
        # 'weights': VGG11_Weights.IMAGENET1K_V1,
        'num_classes': 10,
        'dropout_p': 0.5,
    }

def get_default_model() -> nn.Module:
    params = get_default_model_params()
    return VGG16(**params)