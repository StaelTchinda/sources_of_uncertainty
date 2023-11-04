import warnings
from typing import Any, Dict, Iterable, Optional, List, Text, Tuple, Union

import torch
import torch.nn.functional as F


from .utils.welford import _WelfordAggregate, initialize, update, finalize
from util import verification, assertion

class VarianceEstimator:
    _dataset_size: int
    _existing_aggregates: List[Optional[_WelfordAggregate]]
    _summed_variances: Optional[torch.Tensor]

    def __init__(self, device: Optional[Union[torch.device, Text]] = None):
        self.reset()
        self.device: Optional[torch.device]
        if device is not None:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            self.device = None

    def feed_probs(self,
                   probs: torch.Tensor,
                   sampling_index: int,
                   batch_index: int) -> None:
        if self.device is not None and probs.device != self.device:
            probs = probs.to(self.device)
        elif self.device is None:
            self.device = probs.device

        if sampling_index == 0:
            assertion.assert_equals(batch_index, len(self._existing_aggregates))

            self._existing_aggregates.append(initialize(probs.to(self.device)))
        else:
            # original_cuda_memory = torch.cuda.memory_allocated()
            if self._existing_aggregates[batch_index] is None:
                raise ValueError(f"Batch index {batch_index} has not been initialized or already finalized before.")
            else:
                self._existing_aggregates[batch_index] = update(self._existing_aggregates[batch_index], probs)
            # updated_cuda_memory = torch.cuda.memory_allocated()
            # message = f"Memory consumption with aggregate of shape {self._existing_aggregates[batch_index][1].shape} at count {self._existing_aggregates[batch_index][0]} increased from {original_cuda_memory / 1024 / 1024 / 1024}GB to {updated_cuda_memory / 1024 / 1024 / 1024}GB"
            # if updated_cuda_memory - original_cuda_memory > 10:
                # warnings.warn(message)


    def get_metric_value(self) -> torch.Tensor:
        for i in range(len(self._existing_aggregates)):
            if self._existing_aggregates[i] is not None:
                self.finalize_batch(i)
    
        return self._summed_variances / self._dataset_size
    
    
    def finalize_batch(self, batch_index: int) -> None:
        verification.check_le(batch_index, len(self._existing_aggregates))
        if self._existing_aggregates[batch_index] is not None:
            _variances_by_batch = finalize(self._existing_aggregates[batch_index])[1]
            if self._summed_variances is None:
                self._summed_variances = _variances_by_batch.sum(dim=0)
            else:
                self._summed_variances.add_(_variances_by_batch.sum(dim=0))
            self._dataset_size += _variances_by_batch.shape[0]
            self._existing_aggregates[batch_index] = None
        else:
            raise ValueError(f"Batch index {batch_index} has already been finalized.")
    
    def reset(self) -> None:
        self._dataset_size = 0
        self._existing_aggregates = []
        self._summed_variances = None


