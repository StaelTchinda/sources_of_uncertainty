from typing import Literal, Optional, Union
import torch
import torchmetrics

import numpy as np
from util import assertion

class StandardDev(torchmetrics.Metric):
    is_ensemble_metric: bool = True
    higher_is_better = False

    def __init__(self, top_k: Union[int, Literal['all']]='all'):
        super().__init__()
        self.top_k = top_k
        self.add_state("std_per_samples", default=[], dist_reduce_fx=None)
        # self.add_state("probs", default=[], dist_reduce_fx=None)

    def update(self, probs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assertion.assert_equals(3, len(probs.shape), f"Expected probs to have shape (S, B, C), but got {probs.shape}")
        # print(f"probs shape: {probs.shape}")
        # Take the top_k predictions
        if self.top_k != 'all':
            probs = probs.topk(self.top_k, dim=-1).values
            assertion.assert_equals(self.top_k, probs.size(-1))
            assertion.assert_equals(3, len(probs.shape), f"Expected probs to have shape (S, B, {self.top_k}), but got {probs.shape}")
        std = torch.var(probs, dim=0)
        assertion.assert_equals(probs.shape[1:3], std.shape)
        std = std.mean(dim=-1)
        assertion.assert_equals(probs.shape[1:2], std.shape)
        self.std_per_samples.append(std)
        # self.probs.append(probs)

    def compute(self):
        # probs = torch.cat(self.probs)
        # print(f"Original probs of shape {probs.shape}: {probs}")
        # probs = probs.topk(self.top_k, dim=-1).values if self.top_k != 'all' else probs
        # print(f" filtered to {self.top_k} classes returned probs of shape {probs.shape} : {probs}")
        # std_per_samples = torch.var(probs, dim=0)
        # print(f" which were used to compute the standard deviation per samples over each class of shape {std_per_samples.shape}: {std_per_samples}")
        # std_per_samples = std_per_samples.mean(dim=-1)
        # print(f" which were then averaged over the classes to obtain the std per sample of shape {std_per_samples.shape}: {std_per_samples}")
        std = torch.cat(self.std_per_samples).mean()
        # print(f"Finally obtained std of shape {std.shape}: {std}")
        return std
    

def standard_dev(probs: torch.Tensor, targets: Optional[torch.Tensor] = None, top_k: Union[int, Literal['all']]='all', average: bool = False):
    assertion.assert_equals(3, len(probs.shape), f"Expected probs to have shape (S, B, C), but got {probs.shape}")
    # print(f"probs shape: {probs.shape}")
    # Take the top_k predictions
    if top_k != 'all':
        probs = probs.topk(top_k, dim=-1).values
        assertion.assert_equals(top_k, probs.size(-1))
        assertion.assert_equals(3, len(probs.shape), f"Expected probs to have shape (S, B, {top_k}), but got {probs.shape}")
    std = torch.var(probs, dim=0)
    assertion.assert_equals(probs.shape[1:3], std.shape)
    std = std.mean(dim=-1)
    assertion.assert_equals(probs.shape[1:2], std.shape)
    if average:
        std = std.mean()
    return std