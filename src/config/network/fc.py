from typing import Dict, Text, Any
from torch import nn

from network.architecture.fc import FeedForward


def get_default_model_params() -> Dict[Text, Any]:
    return {
        "in_dim": 4, 
        "out_dim": 3, 
        "hidden_layers": [16, 8], 
        "dropout_probs": [0.2, 0.2]
    }

def get_default_model() -> nn.Module:
    params = get_default_model_params()
    return FeedForward(**params)