from typing import Dict, Text, Any
from torch import nn

from network.architecture.lenet import LeNet5


def get_default_model_params() -> Dict[Text, Any]:
    return {
        'in_channels': 1,
        'num_classes': 10,
        'dropout_p': 0.2,
    }

def get_default_model() -> nn.Module:
    params = get_default_model_params()
    return LeNet5(**params)