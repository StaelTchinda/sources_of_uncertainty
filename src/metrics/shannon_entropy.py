from typing import Literal
import torch
import torchmetrics
import scipy

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
        # self.add_state("probs", default=[], dist_reduce_fx=None)

    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        # self.probs.append(probs)
        if self.dist == 'normal':
            raise NotImplementedError("Normal distribution not implemented")
        elif self.dist == 'categorical':
            entropy = predictive_entropy(probs)
        else:
            raise ValueError(f"Unsupported distribution: {self.dist}")

        self.entropy_per_samples.append(entropy)

    def compute(self):
        # print(f"shannon_entropy - probs of shape {torch.cat(self.probs).shape}: {self.probs}")
        # print(f"shannon_entropy - Entropy per samples of shape {torch.cat(self.entropy_per_samples).shape}: {self.entropy_per_samples}")
        entropy = torch.cat(self.entropy_per_samples).mean()
        return entropy
    


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    :param probs: of shape (*, C), with * is 0 or N, the number of sampled for which the entropy should be computed
    :return:
    """
    assertion.assert_contains([1, 2], len(probs.shape), f"Expected shape (C) or (N, C), but got shape {probs.shape}")
    assertion.assert_not_nan(probs)
    should_reshape: bool = len(probs.shape) == 1
    if should_reshape:
        probs = probs[None, :]
    assertion.assert_tensor_close(torch.ones((probs.size(0))).to(probs.device), probs.sum(dim=1))

    entropy = torch.tensor(scipy.stats.entropy(probs.detach().cpu().numpy(), base=2, axis=1), device=probs.device)

    assertion.assert_not_nan(entropy)

    assertion.assert_equals([probs.size(0)], list(entropy.size()))
    if should_reshape:
        probs = torch.squeeze(probs)
        entropy = torch.squeeze(entropy)
    return entropy
