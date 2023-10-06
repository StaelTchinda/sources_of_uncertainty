from typing import Optional

import torch
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, Wide_ResNet101_2_Weights, ResNet, wide_resnet50_2, wide_resnet101_2
from torchvision.models.resnet import Bottleneck, BasicBlock
from network.bayesian.mc_dropout.mc_dropout import DropoutHook


class WideResNet50(ResNet):
    def __init__(self, num_classes: int = 10, weights: Optional[Wide_ResNet50_2_Weights] = Wide_ResNet50_2_Weights.IMAGENET1K_V2, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            Bottleneck,
            [3, 4, 6, 3], 
            width_per_group=64 * 2)

        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        input_last_layer: int = int(self.fc.in_features)
        self.fc = nn.Linear(input_last_layer, num_classes)

        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts


    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)


class WideResNet101(ResNet):
    def __init__(self, num_classes: int = 10, weights: Optional[Wide_ResNet101_2_Weights] = Wide_ResNet101_2_Weights.IMAGENET1K_V2, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            Bottleneck,
            [3, 4, 23, 3],
            width_per_group=64 * 2)
        
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        input_last_layer: int = int(self.fc.in_features)
        self.fc = nn.Linear(input_last_layer, num_classes)

        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts
    

        
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)
