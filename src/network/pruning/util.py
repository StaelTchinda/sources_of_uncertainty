from typing import List, Dict, Optional, Text, Tuple
import torch
from torch import nn
from util import assertion
from torch.nn.utils import prune
from torch.nn.utils.convert_parameters import parameters_to_vector, parameters_to_vector


def get_prunable_named_modules(model: nn.Module, parameter_name: Text = "weight") -> Dict[Text, nn.Module]:
    # Define which modules of the original model can be pruned
    prunable_modules = {}
    for name, module in model.named_modules():
        if hasattr(module, parameter_name):
            prunable_modules[name] = module
    return prunable_modules


def get_required_amounts_for_sparsities(desired_sparsities: List[float]) -> List[float]:
    required_amounts = []

    current_sparsity = 0.0
    for (i, desired_sparsity) in enumerate(desired_sparsities):
        if i == 0:
            required_amount = desired_sparsity
            current_sparsity = required_amount
        else:
            required_amount = (-current_sparsity + desired_sparsity) / (1 - current_sparsity)
            current_sparsity += (1-current_sparsity) * required_amount
        assertion.assert_equals(current_sparsity, desired_sparsities[i])
        required_amounts.append(required_amount)
        
    return required_amounts

def get_parameter_mask(model: nn.Module, weight: bool = True, bias: bool = True) -> torch.Tensor:
    layer_masks: List[torch.Tensor] = []

    for (layer_name, layer) in model.named_modules():
        layer_parameter_masks: List[torch.Tensor] = []
        if weight:
            if hasattr(layer, 'weight_mask') and isinstance(layer.weight_mask, torch.Tensor): 
                layer_parameter_masks.append(layer.weight_mask.flatten())
            elif hasattr(layer, 'weight') and isinstance(layer.weight, nn.Parameter): 
                layer_parameter_masks.append(torch.ones_like(layer.weight.flatten()))
        if bias:
            if hasattr(layer, 'bias_mask') and isinstance(layer.bias_mask, torch.Tensor): 
                layer_parameter_masks.append(layer.bias_mask.flatten())
            elif hasattr(layer, 'bias') and isinstance(layer.bias, nn.Parameter): 
                layer_parameter_masks.append(torch.ones_like(layer.bias.flatten()))

        if len(layer_parameter_masks) > 0:
            if weight and bias:
                assertion.assert_tensor_same_shape(torch.cat(layer_parameter_masks), parameters_to_vector(layer.parameters()))
            layer_masks.extend(layer_parameter_masks)

    if weight and bias:
        assertion.assert_tensor_same_shape(torch.cat(layer_masks), parameters_to_vector(model.parameters()))

    return torch.cat(layer_masks)



def measure_modular_sparsity(module: nn.Module, weight: bool = True, bias: bool = True, filter_unpruned_modules: bool=False, filter_weightless_modules: bool=True) -> Dict[Text, Tuple[int, int]]:
    result: Dict[Text, Tuple[int, int]] = {}

    module_pruned_count: int = 0
    module_total_count: int = 0

    for (layer_name, layer) in module.named_modules():
        layer_pruned_count: int = 0
        layer_total_count: int = 0 
        if weight and hasattr(layer, 'weight'):
            layer_pruned_count += int(torch.sum(layer.weight == 0).item())
            layer_total_count += layer.weight.nelement()
        if bias and hasattr(layer, 'bias') and isinstance(layer.bias, torch.Tensor):
            layer_pruned_count += int(torch.sum(layer.bias == 0).item())
            layer_total_count += layer.bias.nelement()

        module_pruned_count += layer_pruned_count
        module_total_count += layer_total_count

        # Filter weightless modules
        if filter_weightless_modules and layer_total_count == 0:
            continue      

        # Filter unpruned modules
        if filter_unpruned_modules and layer_pruned_count == 0:
            continue                    
            
        result[layer_name] = (layer_pruned_count, layer_total_count)

    if not filter_unpruned_modules or module_pruned_count > 0:
        result[module._get_name()] = (module_pruned_count, module_total_count)

    return result


def verbose_modular_sparsity(model: nn.Module, modular_sparsity: Optional[Dict[Text, Tuple[int, int]]] = None, weight: bool = True, bias: bool = True, filter_unpruned_layers: bool = True) -> Text:
    if modular_sparsity is None:
        modular_sparsity = measure_modular_sparsity(model, weight, bias, filter_unpruned_modules=False)
    result_text: Text = ""

    model_name: Text = model._get_name()

    for layer_name, (layer_pruned_count, layer_total_count) in modular_sparsity.items():            
        if not filter_unpruned_layers or layer_pruned_count > 0:
            head: Text = "Global sparsity" if layer_name == model_name else f"Sparsity in {layer_name}"
            result_text += f"{head}: {100.0 * layer_pruned_count / layer_total_count:.2f}% = {layer_pruned_count} / {layer_total_count}\n"

    return result_text


def undo_pruning(model: nn.Module):
    for (layer_name, layer) in model.named_modules():
        original_pruned_parameters: Dict[Text, torch.Tensor] = {}

        for (name, parameter) in layer.named_parameters():
            if name.endswith('_orig'):
                original_pruned_parameters[name] = parameter.data.detach().clone()

        for (original_name, original_parameter) in original_pruned_parameters.items():
            usual_name: Text = original_name[:-5]

            prune.remove(layer, usual_name)

            parameter_names: List[Text] = [named_parameter[0] for named_parameter in layer.named_parameters()]
            assertion.assert_contains(parameter_names, usual_name)

            layer.get_parameter(usual_name).data.mul_(0)
            layer.get_parameter(usual_name).data.add_(original_parameter.to(layer.get_parameter(usual_name).device))

    if __debug__:
        modular_sparsity: Dict[Text, Tuple[int, int]] = measure_modular_sparsity(model)
        for (module_name, sparsitiy_counts) in modular_sparsity.items():
            assertion.assert_equals(0, sparsitiy_counts[0], f"Expected at module {module_name} a sparsity of 0, but got {sparsitiy_counts[0]}/{sparsitiy_counts[1]}.")