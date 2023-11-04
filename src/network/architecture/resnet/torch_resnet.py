from typing import Optional, Dict, Text, Literal, Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet, resnet18, resnet34, resnet50
from torchvision.models.resnet import Bottleneck, BasicBlock
from network.bayesian.mc_dropout import DropoutHook, repr_with_dropout_hook

from util import verification


def default_resnet_submodules_to_dropouts(model: nn.Module, p: float, mode: Literal['relu', 'existing'] = 'relu') -> Dict[nn.Module,  nn.modules.dropout._DropoutNd]:
    default_modules: Dict[nn.Module, nn.modules.dropout._DropoutNd] = {}
    prev_module: Optional[nn.Module] = None
    for (module_name, module) in model.named_modules():
        if prev_module is None:
            prev_module = module
            continue

        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            if mode == 'existing' or mode == 'relu':
                default_modules[prev_module] = module
                if module.p != p:
                    warnings.warn(f"Found a dropout layer with probability {module.p} instead of {p}.")
            prev_module = module
            continue

        if not isinstance(module, nn.ReLU):
            prev_module = module
            continue

        if mode == 'relu':
            if isinstance(prev_module, nn.Linear) or isinstance(prev_module, nn.Conv1d) or isinstance(prev_module, nn.BatchNorm1d):
                default_modules[module] = nn.Dropout(p)
            elif isinstance(prev_module, nn.Conv2d) or isinstance(prev_module, nn.BatchNorm2d):
                default_modules[module] = nn.Dropout2d(p)
            else:
                raise ValueError(f"Unexpected module type before a ReLU: {type(prev_module)}")            

    return default_modules

class McDropoutResNet(ResNet):
    def __init__(self, block, layers, num_classes: int =10, weights: Optional[ResNet18_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            block, 
            layers, 
            num_classes=num_classes)
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        if dropout_p is not None:
            self.submodule_to_dropouts = default_resnet_submodules_to_dropouts(self, dropout_p)
            self.dropout_hook = DropoutHook(self, submodule_to_dropouts=self.submodule_to_dropouts)
    
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)
        
    def __repr__(self):
        if hasattr(self, 'dropout_hook'):
            return repr_with_dropout_hook(self, self.dropout_hook)
        else:
            return nn.Module.__repr__(self)
    

class ResNet18(McDropoutResNet):
    def __init__(self, num_classes: int =10, weights: Optional[ResNet18_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            BasicBlock, 
            [2, 2, 2, 2], 
            num_classes=num_classes, 
            weights=weights,
            progress=progress,
            dropout_p=dropout_p)


class ResNet34(McDropoutResNet):
    def __init__(self, num_classes: int =10, weights: Optional[ResNet34_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            BasicBlock, 
            [3, 4, 6, 3], 
            num_classes=num_classes, 
            weights=weights,
            progress=progress,
            dropout_p=dropout_p)


class ResNet50(McDropoutResNet):
    def __init__(self, num_classes: int = 10, weights: Optional[ResNet50_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            Bottleneck,
            [3, 4, 6, 3], 
            num_classes=num_classes, 
            weights=weights,
            progress=progress,
            dropout_p=dropout_p)


class ResNet101(ResNet):
    def __init__(self, num_classes: int = 10, weights: Optional[ResNet101_Weights] = None, progress: bool = False, dropout_p: Optional[float] = None):
        super().__init__(
            Bottleneck,
            [3, 4, 23, 3],
            num_classes=num_classes)

def adapt_resnet_to_cifar10(model: nn.Module):
    verification.check_is_instance(model, ResNet)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model