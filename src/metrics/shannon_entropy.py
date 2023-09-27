from typing import Literal
import torch
import torchmetrics

import numpy as np
from util import assertion

class ShannonEntropy(torchmetrics.Metric):
    is_ensemble_metric: bool = False
    higher_is_better = False

    def __init__(self, dist: Literal['normal', 'categorical']='normal', base: int = 2):
        super().__init__()
        self.dist = dist
        self.base = base
        self.add_state("entropy_per_samples", default=[], dist_reduce_fx=None)
        self.add_state("probs", default=[], dist_reduce_fx=None)

    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        self.probs.append(probs)
        if self.dist == 'normal':
            raise NotImplementedError("Normal distribution not implemented")
        elif self.dist == 'categorical':
            assertion.assert_equals(2, len(probs.shape), f"Expected probs to have shape (N, C), but got {probs.shape}")
            entropy = -torch.sum(probs * torch.log(probs), dim=-1) / torch.log(torch.tensor([self.base], device=probs.device))
            assertion.assert_equals(probs.size(0), entropy.shape[0], f"Expected probs and entropy to have same batch size, but got {probs.size(0)} and {entropy.shape[0]}")
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")

        self.entropy_per_samples.append(entropy)

    def compute(self):
        # print(f"shannon_entropy - probs of shape {torch.cat(self.probs).shape}: {self.probs}")
        # print(f"shannon_entropy - Entropy per samples of shape {torch.cat(self.entropy_per_samples).shape}: {self.entropy_per_samples}")
        entropy = torch.cat(self.entropy_per_samples).mean()
        return entropy