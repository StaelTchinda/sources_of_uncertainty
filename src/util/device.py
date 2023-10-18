from typing import Literal, Optional, Dict, Text, Union

from laplace import ParametricLaplace

import torch
import pytorch_lightning as pl
from torch import nn
import torchmetrics

import laplace as laplace_lib
from laplace import utils as laplace_libutils
import laplace.utils.matrix

from util import verification, assertion



def move_model_to_device(model: nn.Module, device: torch.device) -> torch.device:
    original_device = next(model.parameters()).device
    model.to(device)
    # # TODO: Get rid of this hack fix. It is needed because the model is not moved to the device properly. It did happen when pruning the weights of a submodule.
    # for (_, module) in  model.named_modules():
    #     if hasattr(module, "weight") and module.weight is not None and module.weight.device != device:
    #         module.weight = module.weight.to(device)
    return original_device

def move_laplace_to_device(laplace: laplace_lib.ParametricLaplace, device: Union[torch.device, Text]) -> torch.device:
    if isinstance(device, str):
        device = torch.device(device)

    original_device = laplace._device
    laplace._device = device
    move_model_to_device(laplace.model, device)
    laplace.prior_precision = laplace.prior_precision.to(device)
    if isinstance(laplace, laplace_lib.FullLaplace):
        laplace.H = laplace.H.to(device)
    elif isinstance(laplace, laplace_lib.KronLaplace):
        move_kron_decomposed_to_device(laplace.H, device)
    laplace.mean = laplace.mean.to(device)
    laplace._sigma_noise = laplace._H_factor.to(device)
    return original_device

def move_kron_decomposed_to_device(kron_decomposed: laplace_libutils.KronDecomposed, device: torch.device) -> torch.device:
    original_device = kron_decomposed.eigenvalues[0][0].device
    for (eigvec_idx) in range(len(kron_decomposed.eigenvectors)):
        for (block_idx) in range(len(kron_decomposed.eigenvectors[eigvec_idx])):
            kron_decomposed.eigenvectors[eigvec_idx][block_idx] = kron_decomposed.eigenvectors[eigvec_idx][block_idx].to(device)
    for (eigval_idx) in range(len(kron_decomposed.eigenvalues)):
        for (block_idx) in range(len(kron_decomposed.eigenvalues[eigval_idx])):
            kron_decomposed.eigenvalues[eigval_idx][block_idx] = kron_decomposed.eigenvalues[eigval_idx][block_idx].to(device)
    kron_decomposed.deltas = kron_decomposed.deltas.to(device)
    return original_device