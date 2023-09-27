from typing import List, Optional

import torch
import torch.nn as nn

from util import verification


# Define model
class FeedForward(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: List[int] = [], dropout_probs: List[Optional[float]] = [],
                 bias: bool = True):
        verification.check_equals(len(hidden_layers), len(dropout_probs))
        super(FeedForward, self).__init__()
        linear_relu_layers = []
        layer_in_dim = in_dim
        for i, layer_out_dim in enumerate(hidden_layers):
            linear_relu_layers.append(nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
            linear_relu_layers.append(nn.ReLU())
            if dropout_probs[i] is not None:
                linear_relu_layers.append(nn.Dropout(dropout_probs[i]))
            layer_in_dim = layer_out_dim
        linear_relu_layers.append(nn.Linear(layer_in_dim, out_dim))

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(*linear_relu_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
