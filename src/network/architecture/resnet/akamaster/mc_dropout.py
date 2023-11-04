from typing import Optional, Dict, Text, Literal, Any

import torch
from torch import nn
from torch.nn import functional as F
from network.architecture.resnet.akamaster.original import ResNet, resnet20, resnet32, resnet44, BasicBlock
from network.bayesian.mc_dropout import DropoutHook, repr_with_dropout_hook

from util import verification


def default_resnet_submodules_to_dropouts(model: nn.Module, p: float, mode: Literal['relu', 'existing'] = 'relu') -> Dict[nn.Module,  nn.modules.dropout._DropoutNd]:
    verification.check_is_instance(model, ResNet)

    default_modules: Dict[nn.Module, nn.modules.dropout._DropoutNd] = {}

    default_modules[model.relu] = nn.Dropout(p)

    for res_layer in [model.layer1, model.layer2, model.layer3]:
        for module in res_layer.modules():
            if isinstance(module, nn.ReLU):
                default_modules[module] = nn.Dropout(p)

    return default_modules


class McDropoutBasicBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1, dropout_p: Optional[float] = None):
        super().__init__(in_planes, planes, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class McDropoutResNet(ResNet):
    def __init__(self, block, layers, num_classes: int =10, dropout_p: Optional[float] = None):
        super().__init__(
            block, 
            layers, 
            num_classes=num_classes)

        self.relu = nn.ReLU()
        if dropout_p is not None:
            self.submodule_to_dropouts = default_resnet_submodules_to_dropouts(self, dropout_p)
            self.dropout_hook = DropoutHook(self, submodule_to_dropouts=self.submodule_to_dropouts)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, 'dropout_hook'):
            self.dropout_hook.enable_or_disable(mode)
        
    def __repr__(self):
        if hasattr(self, 'dropout_hook'):
            return repr_with_dropout_hook(self, self.dropout_hook)
        else:
            return nn.Module.__repr__(self)
    

class ResNet20(McDropoutResNet):
    def __init__(self, num_classes: int = 10, dropout_p: Optional[float] = None):
        super().__init__(
            McDropoutBasicBlock, 
            [3, 3, 3], 
            num_classes=num_classes,
            dropout_p=dropout_p)


class ResNet32(McDropoutResNet):
    def __init__(self, num_classes: int = 10, dropout_p: Optional[float] = None):
        super().__init__(
            McDropoutBasicBlock, 
            [5, 5, 5], 
            num_classes=num_classes,
            dropout_p=dropout_p)


class ResNet44(McDropoutResNet):
    def __init__(self, num_classes: int = 10, dropout_p: Optional[float] = None):
        super().__init__(
            McDropoutBasicBlock,
            [7, 7, 7], 
            num_classes=num_classes,
            dropout_p=dropout_p)