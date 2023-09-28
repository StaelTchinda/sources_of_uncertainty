import warnings
from typing import Any, Dict, Iterable, Optional, List, Text, Tuple, Union

import torch
import torch.nn.functional as F


from .utils.welford import _WelfordAggregate, initialize, update, finalize
from .float_metric import FloatMetricEstimator
from util.assertion import assert_contains, assert_equals, assert_le, assert_lt, assert_not_none


class VarianceEstimator(FloatMetricEstimator):
    _batch_sizes: List[int]  # maps from the batch index to the batch size
    _sampling_size: int
    _existing_aggregates: List[_WelfordAggregate]
    _probs_expected_shapes: List[torch.Size]  # maps from the batch index to the expected size of the probs
    _metric_computed: bool

    def __init__(self, warn_mode: bool = True):
        self.reset()
        self.warn_mode: bool = warn_mode
        super().__init__()

    def sampling_size(self) -> int:
        return self._sampling_size

    @classmethod
    # the less, the better
    def compare_values(cls, value1: float, value2: float) -> float:
        return value2 - value1

    @classmethod
    def get_metric_name(cls) -> Text:
        return "variance"


    def feed_probs(self,
                   probs: torch.Tensor,
                   gt_labels: Optional[torch.Tensor] = None,
                   samples: Optional[torch.Tensor] = None,
                   sampling_index: Optional[int] = None,
                   batch_index: Optional[int] = None) -> None:
        feed_params: Dict[Text, Any] = locals()
        FloatMetricEstimator.assert_round_and_batch_index(sampling_index, batch_index)
        assert_contains([1, 2], len(probs.shape), f"Expected shape (C) or (N, C), but got shape {probs.shape}")

        if self.warn_mode and self._metric_computed:
            warnings.warn(f"Probs were fed after the metric value has been computed")

        # Check and update the expected shape of probs
        if sampling_index is None:
            if self._sampling_size == 0:
                assert_equals(0, len(self._probs_expected_shapes))
                self._probs_expected_shapes.append(probs.shape)
            else:
                assert_equals(1, len(self._probs_expected_shapes))
                assert_equals(self._probs_expected_shapes[0], probs.shape)
        else:
            if sampling_index == 0:
                # The batch index should correspond to the next one to be updated
                assert_equals(len(self._probs_expected_shapes), batch_index)
                self._probs_expected_shapes.append(probs.shape)
            else:
                assert_lt(batch_index, len(self._probs_expected_shapes))
                assert_equals(self._probs_expected_shapes[batch_index], probs.shape)

        if sampling_index is None:
            if self._sampling_size == 0:
                self._existing_aggregates.append(initialize(probs))
            else:
                self._existing_aggregates.append(initialize(probs))
        else:
            assert_not_none(batch_index)
            if sampling_index == 0:
                assert_equals(batch_index, len(self._batch_sizes))
                assert_equals(batch_index, len(self._existing_aggregates))
                self._batch_sizes.append(probs.size(0))

                self._existing_aggregates.append(initialize(probs))
            else:
                assert_lt(batch_index, len(self._batch_sizes))
                assert_equals(self._batch_sizes[batch_index], probs.size(0))

                self._existing_aggregates[batch_index] = update(self._existing_aggregates[batch_index], probs)

        if sampling_index is None:
            self._sampling_size += 1
        else:
            if batch_index == 0:
                self._sampling_size += 1
        self.apply_feed_hooks(feed_params)


    def get_metric_value(self, intermediate: bool = False) -> float:
        variances_by_sample = self.get_metric_values_per_samples()
        variance = variances_by_sample.mean(dim=0)
        if not intermediate:
            self._metric_computed = True
        
        # assert_le(0, variance)
        return variance
    
    def get_metric_values_per_samples(self) -> torch.Tensor:
        # of shape(N)
        final_results_by_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [finalize(existing_aggregate) for existing_aggregate in self._existing_aggregates]
        # print(f"final_results: {final_results}")
        means_by_sample: torch.Tensor = torch.cat([batch_final_results[0] for batch_final_results in final_results_by_batch])
        variances_by_sample: torch.Tensor = torch.cat([batch_final_results[1] for batch_final_results in final_results_by_batch])
        sampled_variances_by_sample: torch.Tensor = torch.cat([batch_final_results[2] for batch_final_results in final_results_by_batch])

        return variances_by_sample
    
    def reset(self) -> None:
        self._batch_sizes = []
        self._sampling_size = 0
        self._existing_aggregates = []
        self._probs_expected_shapes = []
        self._metric_computed = False


