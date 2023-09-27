from typing import Optional

import torch
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet, resnet18, resnet34, resnet50
from torchvision.models.resnet import Bottleneck, BasicBlock
from thesis.model.bayesian.mc_dropout.mc_dropout import DropoutHook

from thesis.model.classification_network import ClassificationNetwork


class ResNet18(ResNet):
    def __init__(self, num_classes: int =10, weights: Optional[ResNet18_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            BasicBlock, 
            [2, 2, 2, 2], 
            num_classes=num_classes)
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))
        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts
    
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)
    
class ResNet34(ResNet):
    def __init__(self, num_classes: int =10, weights: Optional[ResNet34_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            BasicBlock, 
            [3, 4, 6, 3], 
            num_classes=num_classes)
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))
        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts
    
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)


class ResNet50(ResNet):
    def __init__(self, num_classes: int = 10, weights: Optional[ResNet50_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            Bottleneck,
            [3, 4, 6, 3], 
            num_classes=num_classes)
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))
        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts


    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)


class ResNet101(ResNet):
    def __init__(self, num_classes: int = 10, weights: Optional[ResNet101_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_classes)
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))
        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts
    

        
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)
