from typing import Dict, List, Literal, Text
import torch
import laplace.utils.matrix

from typing import Callable

from util import assertion
from network import lightning as lightning

NetworkGranularity = Literal['weight', 'channel', 'layer_channel', 'layer']
ReduceMode = Literal['max', 'min', 'mean']

def get_reduce_fn(reduce_mode: ReduceMode) -> Callable[[torch.Tensor], torch.Tensor]:
    if reduce_mode == 'max':
        reduce_fn = torch.max
    elif reduce_mode == 'min':
        reduce_fn = torch.min
    elif reduce_mode == 'mean':
        reduce_fn = torch.mean
    else:
        raise ValueError(f"Unknown reduce function {reduce_mode}")
    
    return reduce_fn

def reduce_eigenvalues_per_channel(weight_eigenvalues: torch.Tensor, model_decomposition: Dict[Text, List[int]], reduce_mode: ReduceMode) -> torch.Tensor:
    reduce_fn = get_reduce_fn(reduce_mode)

    n_channels = sum(len(l) for l in model_decomposition.values())
    n_params = sum(sum(l) for l in model_decomposition.values())
    assertion.assert_equals(n_params, weight_eigenvalues.numel())
    channel_eigenvalues = torch.zeros((n_channels))
    channel_idx = 0
    last_eigval_idx = 0
    for (param_name, n_params_per_channel) in model_decomposition.items():
        for n_channel_params in n_params_per_channel:
            param_eigenvalues = weight_eigenvalues[last_eigval_idx:last_eigval_idx+n_channel_params]
            channel_eigenvalues[channel_idx] = reduce_fn(param_eigenvalues)

            channel_idx += 1
            last_eigval_idx += n_channel_params

    assertion.assert_equals(n_channels, channel_idx)
    assertion.assert_equals(last_eigval_idx, weight_eigenvalues.numel())
    labels = [f'{param_name}.{channel_idx}' for (param_name, channel_param_counts) in model_decomposition.items() for channel_idx in range(len(channel_param_counts))]

    return channel_eigenvalues

def reduce_eigenvalues_per_layer_channel(weight_eigenvalues: torch.Tensor, model_decomposition: Dict[Text, List[int]], reduce_mode: ReduceMode) -> Dict[Text, torch.Tensor]:
    reduce_fn = get_reduce_fn(reduce_mode)

    n_params = sum(sum(l) for l in model_decomposition.values())
    assertion.assert_equals(n_params, weight_eigenvalues.numel())

    channel_eigenvalues = {}
    last_eigval_idx = 0
    for (param_name, n_params_per_channel) in model_decomposition.items():
        channel_eigenvalues[param_name] = torch.zeros((len(n_params_per_channel)))
    
        channel_idx = 0
        for n_channel_params in n_params_per_channel:
            param_eigenvalues = weight_eigenvalues[last_eigval_idx:last_eigval_idx+n_channel_params]
            channel_eigenvalues[param_name][channel_idx] = reduce_fn(param_eigenvalues)

            channel_idx += 1
            last_eigval_idx += n_channel_params

        assertion.assert_equals(len(n_params_per_channel), channel_idx)
    assertion.assert_equals(last_eigval_idx, n_params)

    return channel_eigenvalues    
