
from email.policy import default
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Text, Type
from functools import partial
import warnings

import torch
from torch import nn

from network.bayesian.util import BayesianModuleLike
from util import verification

# FIXME: this function fails with FeedForward
def default_submodules_to_dropouts(model: nn.Module, p: Optional[float], mode: Literal['relu', 'existing'] = 'relu') -> Dict[nn.Module,  nn.modules.dropout._DropoutNd]:
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
                verification.check_not_none(p)
                default_modules[module] = nn.Dropout(p)
            elif isinstance(prev_module, nn.Conv2d) or isinstance(prev_module, nn.BatchNorm2d):
                verification.check_not_none(p)
                default_modules[module] = nn.Dropout2d(p)
            else:
                raise ValueError(f"Unexpected module type before a ReLU: {type(prev_module)}")

        prev_module = module       

    return default_modules



class DropoutHook:
    def __init__(self, model: nn.Module, p: Optional[float]=None, submodule_to_dropouts: Optional[Dict[nn.Module, nn.modules.dropout._DropoutNd]]=None):
        self.model: nn.Module = model
        # self.dropout_prob: float = p

        if submodule_to_dropouts is None:
            self.submodule_to_dropouts: Dict[nn.Module, nn.modules.dropout._DropoutNd] = default_submodules_to_dropouts(self.model, p)
        else:
            self.submodule_to_dropouts = submodule_to_dropouts

        self.check_integrity()
        self.hooks = []
        self.enable_dropout = True

        for (submodule, _) in self.submodule_to_dropouts.items():
            self.hooks.append(submodule.register_forward_hook(self.forward_hook))

    def forward_hook(self, module: nn.Module, args: Dict, output: torch.Tensor) -> torch.Tensor:
        dropout_function: nn.modules.dropout._DropoutNd = self.get_dropout_function(module)
        return dropout_function(output)

    def enable(self):
        self.enable_or_disable(True)

    def disable(self):
        self.enable_or_disable(False)

    def enable_or_disable(self, enable_dropout: bool):
        self.enable_dropout = enable_dropout
        for (submodule, dropout) in self.submodule_to_dropouts.items():
            dropout.train(enable_dropout)

    def remove(self):
        for hook in self.hooks:
            hook.remove()

    def get_dropout_function(self, submodule: nn.Module) -> nn.modules.dropout._DropoutNd:
        if submodule in self.submodule_to_dropouts:
            return self.submodule_to_dropouts[submodule]
        else:
            raise ValueError(f"Submodule {submodule} not found in registered submodules: {self.submodule_to_dropouts}")

    # def init_submodule_to_dropouts(self, submodule_name_to_dropout_methods: Dict[Text, Type[nn.modules.dropout._DropoutNd]]) -> None:
    #     # To avoid creating multiple dropout objects of the same type, we create a dictionary of dropout methods
    #     dropout_method_to_dropout: Dict[Type[nn.modules.dropout._DropoutNd], nn.modules.dropout._DropoutNd] = {}
    #     for (module_name, dropout_method) in submodule_name_to_dropout_methods.items():
    #         submodule = self.model.get_submodule(module_name)
    #         if dropout_method not in dropout_method_to_dropout:
    #             dropout_method_to_dropout[dropout_method] = dropout_method(p=self.dropout_prob)
    #         self.submodule_to_dropouts[submodule] = dropout_method_to_dropout[dropout_method]


    def check_integrity(self) -> None:
        element_checks: Dict[nn.Module, bool] = {submodule: False for submodule in self.submodule_to_dropouts.keys()}

        for (_, submodule) in self.model.named_modules():
            if submodule in self.submodule_to_dropouts:
                element_checks[submodule] = True

        for (submodule, element_check) in element_checks.items():
            if not element_check:
                raise ValueError(f"Expected module {submodule} to belong to the model.")

def verbose_dropout_hook_dict(dropout_hook: DropoutHook) -> Text:
    unseen_modules: List[nn.Module] = list(dropout_hook.submodule_to_dropouts.keys())
    result: Text = ""
    result += (f"Registered submodules in {dropout_hook.model._get_name()}: \n")
    for (module_name, module) in dropout_hook.model.named_modules():
        if module in dropout_hook.submodule_to_dropouts:
            result += (f"\t{module_name}: {module} -> {dropout_hook.submodule_to_dropouts[module]}\n")
            unseen_modules.remove(module)
    
    if len(unseen_modules) > 0:
        result += (f"Modules non-registered in model: \n")
        for module in unseen_modules:
            result += (f"{module} -> {dropout_hook.submodule_to_dropouts[module]}\n")

    return result


# Based on google benchmark, they apply the dropout mask (in ResNet50) after the layers:
# - resnet.BasisBlock.relu1
# - resnet.BasisBlock.relu2
# - resnet.BottleNeck.relu1
# - resnet.BottleNeck.relu2
# - resnet.BottleNeck.relu3
# - resnet.relu1
# The default dropout rate is 0.1
def make_module_bayesian_like(model: nn.Module, dropout_hook: DropoutHook, sampling_size: int) -> BayesianModuleLike:

    def bayesian_like_function(x: torch.Tensor) -> Iterable[torch.Tensor]:
        dropout_hook.enable()
        for i in range(sampling_size):
            yield model(x)
        dropout_hook.disable()

    return bayesian_like_function

