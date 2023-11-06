from typing import Dict, List, Text
import torch
from torch import nn
import laplace.utils.matrix


from util import assertion
from network import lightning as lightning




def compute_model_decomposition(model: nn.Module, strict: bool = False) -> Dict[Text, List[int]]:
    # Given a module, computes a dictionary with keys the names of the layers, and values a list of the number of parameters per channel
    decomposition: Dict[Text, List[int]] = {}
    for (name, layer) in model.named_modules():
        if hasattr(layer, 'weight'):
            if len(layer.weight.shape) == 1:
                decomposition[f"{name}.weight"] = [layer.weight.shape[0]]
            else:
            # assertion.assert_le(2, len(layer.weight.shape), f"Expected layer {name} to have a weight of shape (C, ...), but got {layer.weight.shape}")
                n_params_per_channel: int = int(torch.prod(torch.tensor(layer.weight.shape[1:])).item())
                decomposition[f"{name}.weight"] = [n_params_per_channel] * layer.weight.shape[0]
        if hasattr(layer, 'bias') and layer.bias is not None:
            decomposition[f"{name}.bias"] = [1] * layer.bias.shape[0]	    
    # Verify if the sum of all the parameters is equal to the number of parameters of the model
    if strict:
        assertion.assert_equals(sum(sum(l) for l in decomposition.values()), sum(p.numel() for p in model.parameters()))

    return decomposition
