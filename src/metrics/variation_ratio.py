from typing import Literal, Optional, Union
import torch
from torch.nn import functional as F
import torchmetrics

import numpy as np
from util import assertion

class VariationRatio(torchmetrics.Metric):
    is_ensemble_metric: bool = True
    higher_is_better = False

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("f_x", default=[], dist_reduce_fx=None)
        self.add_state("sampling_size", default=torch.tensor([]), dist_reduce_fx=None)
        # self.add_state("probs", default=[], dist_reduce_fx=None)

    def update(self, probs: torch.Tensor, targets: Optional[torch.Tensor] = None):
        assertion.assert_equals(3, len(probs.shape), f"Expected probs to have shape (S, B, C), but got {probs.shape}")
        # print(f"probs shape: {probs.shape}")

        if self.sampling_size.numel() == 0:
            self.sampling_size = torch.tensor(probs.size(0))
        else:
            assertion.assert_equals(probs.size(0), self.sampling_size.item())

        # of shape (S, B, )
        predicted_labels: torch.Tensor = probs.argmax(dim=-1)

        # of shape (S, B, C)
        one_hot_vector: torch.Tensor = F.one_hot(predicted_labels, num_classes=self.num_classes)

        # of shape (B, C)
        class_pred_occurrences = one_hot_vector.sum(dim=0)

        # of shape (B, )
        f_x: torch.Tensor = class_pred_occurrences.max(dim=1).values

        self.f_x.append(f_x)
        # self.probs.append(probs)

    def compute(self):
        # probs = torch.cat(self.probs)
        # print(f"Original probs of shape {probs.shape}: {probs}")
        
        # of shape(\Sigma B_i)
        f_x: torch.Tensor = torch.cat(self.f_x)
        variation_ratio: torch.Tensor = (1.0 - (f_x / self.sampling_size))

        return variation_ratio.mean()
    
