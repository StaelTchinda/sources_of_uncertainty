from typing import Any, Dict, Literal, Text, Union, Optional, Callable
import logging
import os
import resource

import scipy
import scipy.stats
import random
import numpy as np
import torch
import torch.backends.cudnn
from torch import distributions as dists
from torch.utils import data as torch_data
from torch import nn
import laplace
from laplace import ParametricLaplace
from netcal import metrics as netcal

from util import assertion, lightning as lightning_util, verification



@torch.no_grad()
def predict(dataloader: torch_data.DataLoader, model: Union[nn.Module, laplace.ParametricLaplace], pred_type: Literal['glm', 'nn'] = None, n_samples: int = None):
    py = []
    if isinstance(model, nn.Module):
        device = next(model.parameters()).device
    elif isinstance(model, ParametricLaplace):
        device = model._device
    else:
        raise ValueError(f"Unknown model type {type(model)}") 

    try:
        for batch in dataloader:
            x = batch[0]
            if isinstance(model, ParametricLaplace):
                verification.check_not_none(pred_type)
                verification.check_not_none(n_samples)
                probs = model(x.to(device), pred_type=pred_type, link_approx="mc", n_samples=n_samples)
            else:
                probs = model(x.to(device))
            py.append(probs)
    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()
        raise e
    return torch.cat(py).cpu()

def evaluate_model(model_: Union[nn.Module, laplace.ParametricLaplace], data_loader: torch_data.DataLoader, prefix: Text, pred_type: Literal['glm', 'nn'] = None, n_samples: int = None):
    probs = predict(data_loader, model_, pred_type=pred_type, n_samples=n_samples)
    targets = torch.cat([batch[1] for batch in data_loader], dim=0)
    # if probs.numel() < 1000:
    #     print(f"probs: {probs}, \n preds: {probs.argmax(-1)}, \n targets: {targets}")
    acc = (probs.argmax(-1) == targets).float().mean()
    ece = netcal.ECE(bins=15).measure(probs.numpy(), targets.numpy())
    nll = -dists.Categorical(probs).log_prob(targets).mean()
    assertion.assert_equals(2, len(probs.shape), f"Expected probs to have shape (N, C), but got {probs.shape}")
    # print(f"evaluate_model - probs of shape {probs.shape}: {probs}")
    samples_entropy = scipy.stats.entropy(probs.numpy(), base=2, axis=-1)
    assertion.assert_equals(probs.size(0), samples_entropy.shape[0], f"Expected probs and samples_entropy to have same batch size, but got {probs.size(0)} and {samples_entropy.shape[0]}")
    # print(f"evaluate_model - Entropy per samples of shape {samples_entropy.shape}: {samples_entropy}")
    entropy = samples_entropy.mean()
    # print(f"Entropy: {entropy}")

    print(f'[{prefix}] Acc.: {acc:.2%}; ECE: {ece:.2%}; NLL: {nll:.3}; Entropy: {entropy:.3}')
    return acc, ece, nll


def verbose_and_log(message: Text, verbose: bool = False, log: bool = False, log_level: int = logging.INFO, main_process_check: bool = True):
    if main_process_check and not lightning_util.is_main_ddp_process():
        return
    if verbose:
        print(message)
    if log:
        logging.log(log_level, message)

def get_required_amounts_for_sparsities(desired_sparsities):
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


def cleanup():
    # Add your cleanup code here, such as closing files, releasing resources, etc.
    print("Cleaning up before exit...")
    # Terminate any child processes if needed
    os.system("pkill -P {}".format(os.getpid()))

def register_cleanup():
    import atexit
    atexit.register(cleanup)

def make_deterministic(mode: bool=True, seed: int = 7777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = not mode
    torch.use_deterministic_algorithms(mode, warn_only=True)


def catch_and_print(func: Callable[[], Any]) -> Any:
    try:
        return func()
    except Exception as e:
        # Handle exceptions gracefully, log errors, etc.
        print("An error occurred:", str(e))
        # Print stacktrace
        import traceback
        traceback.print_exc()   

# In case of very slow code, to safely check for bottlenecks in limited time
# from test import bottleneck
# output_path=f"analyse_layer_bottleneck_{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.txt"
# bottleneck.profile_and_stats(lambda: utils.catch_and_print(lambda: utils.run_with_timeout(main, timeout=40*60)), output_path=output_path)
    
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run_with_timeout(func, args=(), kwargs={}, timeout=120):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    result = func(*args, **kwargs)
    return result


def limit_memory(memory_bytes: int):
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    # print('Soft limit starts as  :', soft)
    resource.setrlimit(rsrc, (memory_bytes, hard))
    # soft, hard = resource.getrlimit(rsrc)
    # print('Soft limit changed to :', soft)