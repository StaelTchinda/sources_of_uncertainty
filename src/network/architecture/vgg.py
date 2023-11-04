from typing import Dict, Optional

import torch
from torch import nn
from torchvision.models._api import WeightsEnum
from torchvision.models import VGG11_Weights, VGG13_Weights, VGG16_Weights, VGG16_BN_Weights, VGG, vgg16, vgg11
from torchvision.models.vgg import make_layers, cfgs
from torchvision.models.resnet import Bottleneck

from network.bayesian.mc_dropout import DropoutHook
from util import verification, assertion
import warnings


def default_vgg_submodules_to_dropouts(model: VGG, p: float) -> Dict[nn.Module,  nn.modules.dropout._DropoutNd]:
    verification.check_is_instance(model, VGG)
    default_modules: Dict[nn.Module, nn.modules.dropout._DropoutNd] = {}

    for (module_name, module) in model.features.named_modules():
        if isinstance(module, nn.ReLU):
            default_modules[module] = nn.Dropout2d(p)

    
    prev_module: Optional[nn.Module] = None
    for (module_name, module) in model.classifier.named_modules():
        if prev_module is None:
            prev_module = module
            continue

        if isinstance(module, nn.Dropout):
            assertion.assert_is_instance(prev_module, nn.ReLU)
            default_modules[prev_module] = module
            if module.p != p:
                warnings.warn(f"Found a dropout layer with probability {module.p} instead of {p}.")

        prev_module = module       

    return default_modules


class VggMcDropout(VGG): 
    def __init__(self, default_make_layers_params: Dict, num_classes: int =10, weights: Optional[WeightsEnum] = None, progress: bool = False, dropout_p: float = 0.5):
        super().__init__(
            make_layers(**default_make_layers_params),
            dropout=dropout_p
        )
        if weights:
            self.load_state_dict(weights.get_state_dict(progress=progress))

        input_last_layer: int = int(self.classifier[6].in_features)
        self.classifier[6] = nn.Linear(input_last_layer, num_classes)

        if dropout_p is not None:
            self.submodule_to_dropouts = default_vgg_submodules_to_dropouts(self, dropout_p)
            self.dropout_hook = DropoutHook(self, submodule_to_dropouts=self.submodule_to_dropouts)

    def train(self, mode: bool = True):
        if hasattr(self, "dropout_hook"):
            self.dropout_hook.enable_or_disable(mode)
        super().train(mode)


class VGG16(VggMcDropout):
    def __init__(self, num_classes: int =10, weights: Optional[VGG16_Weights] = VGG16_Weights.IMAGENET1K_V1, progress: bool = False, dropout_p: float = 0.5):
        default_make_layers_params: Dict = {
            'cfg': cfgs["D"],
            'batch_norm': False
        }
        super().__init__(
            default_make_layers_params, 
            num_classes=num_classes,
            weights=weights,
            progress=progress, 
            dropout_p=dropout_p
        )
    
class VGG11(VggMcDropout):
    def __init__(self, num_classes: int =10, weights: Optional[VGG11_Weights] = VGG11_Weights.IMAGENET1K_V1, progress: bool = False, dropout_p: float = 0.5):
        default_make_layers_params: Dict = {
            'cfg': cfgs["A"],
            'batch_norm': False
        }
        super().__init__(
            default_make_layers_params, 
            num_classes=num_classes,
            weights=weights,
            progress=progress, 
            dropout_p=dropout_p
        )    

class VGG13(VggMcDropout):
    def __init__(self, num_classes: int =10, weights: Optional[VGG13_Weights] = VGG13_Weights.IMAGENET1K_V1, progress: bool = False, dropout_p: float = 0.5):
        default_make_layers_params: Dict = {
            'cfg': cfgs["B"],
            'batch_norm': False
        }
        super().__init__(
            default_make_layers_params, 
            num_classes=num_classes,
            weights=weights,
            progress=progress, 
            dropout_p=dropout_p
        )
