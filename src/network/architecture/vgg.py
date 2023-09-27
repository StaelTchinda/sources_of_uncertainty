from typing import Dict, Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet, resnet50
from torchvision.models import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG16_BN_Weights, VGG, vgg16, vgg11
from torchvision.models.vgg import make_layers, cfgs
from torchvision.models.resnet import Bottleneck

from network.bayesian.mc_dropout.mc_dropout import DropoutHook



class VGG16(VGG):
    def __init__(self, num_classes: int =10, weights: Optional[VGG16_Weights] = VGG16_Weights.IMAGENET1K_V1, progress: bool = False, dropout_p: Optional[float] = None):
        default_make_layers_params: Dict = {
            'cfg': cfgs["D"],
            'batch_norm': False
        }
        super().__init__(
            make_layers(**default_make_layers_params)
        )
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        input_last_layer: int = int(self.classifier[6].in_features)
        self.classifier[6] = nn.Linear(input_last_layer, num_classes)

        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts

class VGG11(VGG):
    def __init__(self, num_classes: int =10, weights: Optional[VGG11_Weights] = VGG11_Weights.IMAGENET1K_V1, progress: bool = False, dropout_p: Optional[float] = None):
        default_make_layers_params: Dict = {
            'cfg': cfgs["A"],
            'batch_norm': False
        }
        super().__init__(
            make_layers(**default_make_layers_params)
        )
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        input_last_layer: int = int(self.classifier[6].in_features)
        self.classifier[6] = nn.Linear(input_last_layer, num_classes)

        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts
    

class VGG13(VGG):
    def __init__(self, num_classes: int =10, weights: Optional[VGG13_Weights] = VGG13_Weights.IMAGENET1K_V1, progress: bool = False, dropout_p: Optional[float] = None):
        default_make_layers_params: Dict = {
            'cfg': cfgs["B"],
            'batch_norm': False
        }
        super().__init__(
            make_layers(**default_make_layers_params)
        )
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        input_last_layer: int = int(self.classifier[6].in_features)
        self.classifier[6] = nn.Linear(input_last_layer, num_classes)

        if dropout_p is not None:
            self.dropout_hook = DropoutHook(self, dropout_p)
            self.submodule_to_dropouts = self.dropout_hook.submodule_to_dropouts
