from typing import Any, Callable, Dict, List, Optional, Text

import torch
from torch import nn
from torch.nn import functional as F
from network.bayesian.mc_dropout import DropoutHook



class LeNet5(nn.Module):
    _original_activation_function: Callable[[], nn.Module] = nn.Sigmoid  # nn.ReLU
    _original_pool_operation: Callable[[Any], nn.Module] = nn.AvgPool2d  # nn.MaxPool2d

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 activation_function: Callable[[], nn.Module] = nn.ReLU,
                 pool_operation: Callable[[Any], nn.Module] = nn.MaxPool2d,
                 dropout_p: Optional[float] = None):
        super(LeNet5, self).__init__()

        # To justify why I apply dropout after the activation function
        # https://stats.stackexchange.com/a/317313
        # https://stats.stackexchange.com/a/445233
        optional_dropout2d_layer: Optional[nn.modules.dropout._DropoutNd] = nn.Dropout2d(p=dropout_p) if (dropout_p is not None) else None
        get_optional_dropout2d_layer: Callable[[], List[nn.modules.dropout._DropoutNd]] = lambda: [optional_dropout2d_layer] if optional_dropout2d_layer is not None else []
        
        feature_extractor_layers: List[nn.Module] = [
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            activation_function()] + \
            get_optional_dropout2d_layer() + [
            pool_operation(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            activation_function()] + \
            get_optional_dropout2d_layer() + [
            pool_operation(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            activation_function()] + \
            get_optional_dropout2d_layer()

        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        optional_dropout_layer: Optional[nn.modules.dropout._DropoutNd] = nn.Dropout(p=dropout_p) if (dropout_p is not None) else None
        get_optional_dropout_layer: Callable[[], List[nn.modules.dropout._DropoutNd]] = lambda: [optional_dropout_layer] if optional_dropout_layer is not None else []
        classifier_layers: List[nn.Module] = [
            nn.Linear(in_features=120, out_features=84)] + \
            get_optional_dropout_layer() + [
            activation_function(),
            nn.Linear(in_features=84, out_features=num_classes)]
        
        self.classifier = nn.Sequential(*classifier_layers)

        if dropout_p is not None:
            if optional_dropout2d_layer is None:
                raise ValueError("optional_dropout2d_layer must not be None if dropout_p is not None.")
            if optional_dropout_layer is None:
                raise ValueError("optional_dropout_layer must not be None if dropout_p is not None.")
            self.submodule_to_dropouts: Dict[nn.Module, nn.modules.dropout._DropoutNd] = {
                feature_extractor_layers[1]: optional_dropout2d_layer,
                feature_extractor_layers[5]: optional_dropout2d_layer,
                feature_extractor_layers[9]: optional_dropout2d_layer,
                classifier_layers[0]: optional_dropout_layer
            }
            self.dropout_hook = DropoutHook(self, dropout_p, submodule_to_dropouts=self.submodule_to_dropouts)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


