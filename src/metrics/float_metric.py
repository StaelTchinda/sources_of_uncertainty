from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Text, Tuple

import torch
from torch.utils import hooks


from metrics.abstract_metric import AbstractMetricEstimator
from util.assertion import assert_contains_all, assert_lt, assert_not_none, assert_equals, assert_contains, assert_tensor_same_shape


class FloatMetricEstimator(AbstractMetricEstimator[float]):
    _get_per_samples_hooks: Dict[int, Callable]
    
    def __init__(self):
        self._get_per_samples_hooks = OrderedDict()
        super().__init__()

    @abstractmethod
    def get_metric_values_per_samples(self) -> torch.Tensor:
        pass

    def register_get_per_samples_hook(self, hook: Callable[[FloatMetricEstimator, torch.Tensor], Optional[torch.Tensor]], prepend: bool = False) -> hooks.RemovableHandle:
        handle = hooks.RemovableHandle(
            self._get_per_samples_hooks
        )
        self._get_per_samples_hooks[handle.id] = hook

        if prepend:
            self._get_per_samples_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def apply_get_per_samples_hooks(self, metric_per_values: torch.Tensor) -> Optional[torch.Tensor]:
        output: Optional[torch.Tensor] = None
        tmp_output: Optional[torch.Tensor] = None
        for (id, get_per_samples_hook) in self._get_per_samples_hooks.items():
            if output is not None:
                tmp_output = get_per_samples_hook(self, output)
            else:
                tmp_output = get_per_samples_hook(self, metric_per_values)

            if tmp_output is not None:
                output = tmp_output

        return output
    
    @classmethod
    def unwrap_feed_hook_params(cls, kwargs: Dict[Text, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], Optional[int]] :
            # Extract feed params
        assert_contains_all(list(kwargs.keys()), ['probs', 'gt_labels', 'samples', 'sampling_index', 'batch_index'])
        probs: torch.Tensor = kwargs['probs']
        gt_labels: Optional[torch.Tensor] = kwargs['gt_labels']
        samples: Optional[torch.Tensor] = kwargs['samples']
        sampling_index: Optional[int] = kwargs['sampling_index']
        batch_index: Optional[int] = kwargs['batch_index']
        # assert_not_none(samples)
        # assert_not_none(gt_labels)
        return probs, gt_labels, samples, sampling_index, batch_index


class BatchShapeAsserter():
    _probs_expected_shapes: List[torch.Size]  # maps from the batch index to the expected size of the probs
    _should_reshape: bool
    _sampling_size: int
    def __init__(self) -> None:
        self.reset()

    def update_sampling_size(self, sampling_size: int):
        self._sampling_size = sampling_size

    def check_expected_shape(self,
                            probs: torch.Tensor,
                            sampling_index: Optional[int] = None,
                            batch_index: Optional[int] = None) -> None:
        # Check and update the expected shape of probs
        if sampling_index is None:
            if self._sampling_size == 0:
                assert_equals(0, len(self._probs_expected_shapes))
                self._probs_expected_shapes.append(probs.shape)
                self._should_reshape = len(probs.shape) == 1
            else:
                assert_equals(1, len(self._probs_expected_shapes))
                assert_equals(self._probs_expected_shapes[0], probs.shape)
                assert_equals(self._should_reshape, len(probs.shape) == 1)

        else:
            if sampling_index == 0:
                # The batch index should correspond to the next one to be updated
                assert_equals(len(self._probs_expected_shapes), batch_index)
                self._probs_expected_shapes.append(probs.shape)
                if batch_index == 0:
                    self._should_reshape = len(probs.shape) == 1
                else:
                    assert_equals(self._should_reshape, len(probs.shape) == 1)
            else:
                assert_lt(batch_index, len(self._probs_expected_shapes))
                assert_equals(self._probs_expected_shapes[batch_index], probs.shape)
                assert_equals(self._should_reshape, len(probs.shape) == 1)

    def get_expected_batch_shape(self, batch_index: int) -> torch.Size:
        assert_lt(batch_index, len(self._probs_expected_shapes))
        return self._probs_expected_shapes[batch_index]


    def get_expected_batch_mask(self, batch_index: int) -> torch.Tensor:
        """Get the mask of a batch if we assume all the batches are concatenated

        Args:
            batch_index (int): index of a batch
        """
        assert_lt(batch_index, len(self._probs_expected_shapes))
        batch_mask = torch.tensor([], dtype=torch.bool)
        for (i, batch_shape) in enumerate(self._probs_expected_shapes):
            partial_batch_mask = torch.ones((batch_shape[0],)) if i == batch_index else torch.zeros((batch_shape[0],))
            partial_batch_mask = partial_batch_mask.to(dtype=torch.bool)
            batch_mask = torch.cat((batch_mask, partial_batch_mask))
        return batch_mask

    def get_expected_batch_indices(self, batch_index: int) -> torch.Tensor:
        """Get the mask of a batch if we assume all the batches are concatenated

        Args:
            batch_index (int): index of a batch
        """
        assert_lt(batch_index, len(self._probs_expected_shapes))
        start_index: int = 0
        end_index: int = 0
        for (i, batch_shape) in enumerate(self._probs_expected_shapes):
            if i < batch_index:
                start_index += batch_shape[0]
                end_index += batch_shape[0]
            if i==batch_index:
                end_index += batch_shape[0]
        batch_indices = torch.arange(start_index, end_index)
        return batch_indices


    def reset(self) -> None:
        self._sampling_size = 0
        self._probs_expected_shapes = []

