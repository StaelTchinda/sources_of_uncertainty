from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, List, Optional, OrderedDict, Text, Tuple, TypeVar, Union

import torch
from torch.utils import hooks

from util.assertion import assert_not_none, assert_equals, assert_contains


T = TypeVar('T')

class AbstractMetricEstimator(ABC, Generic[T]):
    _forward_hooks: Dict[int, Callable]

    def __init__(self) -> None:
        self._forward_hooks = OrderedDict()

    # @property
    @abstractmethod
    def sampling_size(self) -> int:
        pass

    @abstractmethod
    def feed_probs(self,
                   probs: torch.Tensor,
                   gt_labels: Optional[torch.Tensor] = None,
                   samples: Optional[torch.Tensor] = None,
                   sampling_index: Optional[int] = None,
                   batch_index: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def get_metric_value(self, intermediate: bool = False) -> T:
        """
        intermediate: bool -> if True, the method would not consider the metric as computed 
        per_sample: bool -> if True, the method would return the metric value for each sample 
        """
        pass

    @classmethod
    @abstractmethod
    def get_metric_name(cls) -> Text:
        pass 

    @abstractmethod
    def reset(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def compare_values(cls, value1: T, value2: T) -> float:
        pass

    @classmethod
    def assert_round_and_batch_index(cls, sampling_index: Optional[int], batch_index: Optional[int],
                                     last_sampling_index: Optional[int] = None, last_batch_index: Optional[int] = None):
        if not __debug__:
            return
        if last_sampling_index is not None:
            assert_not_none(sampling_index, f"current sampling_index is required, if it has been given one time before.")
            # assert_le(last_sampling_index, sampling_index)
        if last_batch_index is not None:
            assert_not_none(batch_index, f"current batch_index is required, if it has been given one time before.")
            # assert_le(last_batch_index, batch_index)
        assert_equals(sampling_index is None, batch_index is None,
                      f"sampling_index and batch_index must be both specified if one of them is. But got: { {'sampling_index': sampling_index, 'batch_index': batch_index} }")

    
    def register_pre_feed_hook(self, hook: Union[
                                            Callable[[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]], bool],
                                            Callable[[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]], Tuple[torch.Tensor, Optional[torch.Tensor]]]], prepend=False):
        """Register a pre-feed hook. A pre-feed hook is operated on the input received. It can alter the input which will be given to feed_probs or just skip this run of the function if it returns False

        Args:
            hook (Callable[[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[int]], Union[bool, Tuple[torch.Tensor, Optional[torch.Tensor]]]]): _description_
            prepend (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        raise NotImplementedError
        

    def register_feed_hook(self, hook: Callable[[AbstractMetricEstimator, Dict[Text, Any]], None], prepend: bool = False) -> hooks.RemovableHandle:
        handle = hooks.RemovableHandle(
            self._forward_hooks
        )
        self._forward_hooks[handle.id] = hook

        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle


    def apply_feed_hooks(self, kwargs: Dict[Text, Any]):
        for (id, forward_hook) in self._forward_hooks.items():
            forward_hook(self, kwargs)




def assert_probs_and_gt_labels_shape(probs: torch.Tensor, gt_labels: torch.Tensor):
    assert_contains([(1, 0), (2, 1)], (len(probs.shape), len(gt_labels.shape)),
                    f"Expected shapes (*, C) resp. (*) from probs resp. gt_labels, "
                    f"but got shapes {probs.shape} resp. {gt_labels.shape}.")
