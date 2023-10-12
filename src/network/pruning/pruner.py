
from typing import Dict, Literal, Optional, Text, Union
import pytorch_lightning as pl


import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.utils.prune as prune
import torchmetrics
from network.lightning import laplace as lightning_laplace
from network.pruning.util import get_parameter_mask

from util import assertion, verification, device as device_util

AVAILABLE_PRUNING_STRATEGIES = ["l1_norm", "l2_norm", "random"]
PruningStrategy = Literal["l1_norm", "l2_norm", "random"] 

class Pruner:
    def __init__(self, original_model: nn.Module,
                 module_to_prune: Optional[Text] = None, pruning_strategy: Optional[PruningStrategy] = None, pruning_amount: Optional[float] = None, pruning_sparsity: Optional[float] = None, pruning_parameter: Text = "weight"):
        super(Pruner, self).__init__()
        self.original_model = original_model
        self.module_name_to_prune = module_to_prune
        self.pruning_strategy:Optional[PruningStrategy] = pruning_strategy
        self.pruning_amount = pruning_amount
        self.pruning_sparsity = pruning_sparsity
        self.pruning_parameter = pruning_parameter


    @property
    def module_to_prune(self) -> Optional[nn.Module]:
        if self.module_name_to_prune is None:
            return None
        else:
            return get_named_module(self.original_model, self.module_name_to_prune)

    def prune(self):
        # Apply pruning based on the prior settings
        verification.check_not_none(self.module_name_to_prune)
        verification.check_not_none(self.pruning_strategy)
        verification.check_not_none(self.pruning_amount)
        verification.check_not_none(self.pruning_parameter)    
        module = get_named_module(self.original_model, self.module_name_to_prune)

        # Pruning can only be done on cpu
        module_device = next(module.parameters()).device
        cpu_device = torch.device('cpu')
        if module_device != cpu_device:
            original_device = device_util.move_model_to_device(module, cpu_device)
        # TODO: Get rid of this hack fix. It is needed because the model is not moved to the device properly. It did happen when pruning the weights of a submodule.
        if hasattr(module, self.pruning_parameter) and \
            getattr(module, self.pruning_parameter) is not None and \
            getattr(module, self.pruning_parameter).device != cpu_device:
                setattr(module, self.pruning_parameter, getattr(module, self.pruning_parameter).to(cpu_device))

        prune_based_on_strategy(module, strategy=self.pruning_strategy, amount=self.pruning_amount, parameter=self.pruning_parameter)

        if module_device != cpu_device:
            device_util.move_model_to_device(module, original_device)

def get_named_module(model: nn.Module, module_name: Text) -> nn.Module:
    return dict(model.named_modules())[module_name]


def prune_based_on_strategy(model: nn.Module, strategy: PruningStrategy, amount: Union[float, int], parameter: Text = "weight", dim: int = 0):
    if strategy=='l1_norm':
        prune.ln_structured(model, parameter, amount, n=1, dim=dim)
    elif strategy=='l2_norm':
        prune.ln_structured(model, parameter, amount, n=2, dim=dim)
    elif strategy=='random':
        prune.random_structured(model, parameter, amount, dim=dim)
    else:
        raise ValueError(f"Invalid strategy: '{strategy}'")