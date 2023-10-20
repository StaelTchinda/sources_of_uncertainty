from typing import Literal, Optional, Union
import torch
import torchmetrics

import numpy as np
from metrics.shannon_entropy import predictive_entropy
from util import assertion

class MutualInformation(torchmetrics.Metric):
    is_ensemble_metric: bool = True
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("summed_probs", default=[], dist_reduce_fx=None)
        self.add_state("summed_entropies", default=[], dist_reduce_fx=None)
        self.add_state("sampling_size", default=torch.tensor([]), dist_reduce_fx=None)
        # self.add_state("probs", default=[], dist_reduce_fx=None)

    def update(self, probs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assertion.assert_equals(3, len(probs.shape), f"Expected probs to have shape (S, B, C), but got {probs.shape}")
        # print(f"probs shape: {probs.shape}")
        if self.sampling_size.numel() == 0:
            self.sampling_size = torch.tensor(probs.size(0))
        else:
            assertion.assert_equals(probs.size(0), self.sampling_size.item())
        self.summed_probs.append(probs.sum(dim=0))
        # IDEA: linearize operation
        self.summed_entropies.append(torch.stack([predictive_entropy(sampled_probs) for sampled_probs in probs]).sum(dim=0))

    def compute(self):
        # probs = torch.cat(self.probs)
        # print(f"Original probs of shape {probs.shape}: {probs}")

        # of shape (N, C)
        p_hat: torch.Tensor = torch.cat(self.summed_probs) / self.sampling_size
        # assert_equals([N, C], list(p_hat.shape))
        
        # of shape (N,)
        entropy_p_hat: torch.Tensor = predictive_entropy(p_hat.detach().clone())
        # assert_equals([N], list(entropy_p_hat.shape))

        # of shape (N,)
        entropies_p: torch.Tensor = torch.cat(self.summed_entropies) / self.sampling_size
        # assert_equals([N], list(entropies_p.shape))

        mutual_information: torch.Tensor = (entropy_p_hat - entropies_p)

        return mutual_information.mean()
    
