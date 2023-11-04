from typing import Dict, Text, Any
from torch import nn

from network.architecture.resnet.torch_resnet import ResNet18, adapt_resnet_to_cifar10


def get_default_model_params() -> Dict[Text, Any]:
    return {
        'num_classes': 10,
        'dropout_p': 0.5,
    }

def get_default_model() -> nn.Module:
    params = get_default_model_params()
    model = ResNet18(**params)
    # TODO: make this configurable
    adapt_resnet_to_cifar10(model)
    return model
    