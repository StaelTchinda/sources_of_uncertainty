# from abc import ABC
# from typing import Any, List, Optional
# import warnings
# import torch

# import torchmetrics

# from metrics.utils import welford
# from util import verification


# class ProbsAccumulator():
#     _sampling_size: Optional[int]
#     # organized by batch
#     _accumulated_preds: List[torch.Tensor] 
#     _variances: List[torch.Tensor]
#     _targets: List[torch.Tensor]

#     main_feeder: Optional[torchmetrics.Metric]

#     def __init__(self, warn_mode: bool = True, main_feeder: Optional[torchmetrics.Metric] = None) -> None:
#         self.warn_mode: bool = warn_mode
#         self.main_feeder = main_feeder
#         self.reset()
#         super().__init__()

#     def sampling_size(self) -> int:
#         verification.check_not_none(self._sampling_size)
#         return self._sampling_size
    
#     def _check_input_shape(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
#         verification.check_equals(len(preds.shape), len(targets.shape)+1, 
#                                   f"Expected predictions to have one dimension more than targets because of sampling, but got predictions of shape {preds.shape} and targets of shape {targets.shape}")
#         if self._sampling_size is not None:
#             verification.check_equals(self._sampling_size, preds.shape[0])
#         # IDEA: add further checks

#     def update(self,
#                 preds: torch.Tensor,
#                 targets: torch.Tensor,
#                 feeder: Optional[torchmetrics.Metric] = None) -> None:
#         if self.main_feeder is not None:
#             if feeder is None:
#                 raise ValueError(f"Feeder must be specified when main_feeder is not None")
#             elif feeder != self.main_feeder:
#                 raise ValueError(f"Feeder {feeder} is not the main_feeder {self.main_feeder}")
#         elif self.warn_mode and feeder is not None:
#             warnings.warn(f"main_feeder is not set")

#         self._check_input_shape(preds, targets)
#         if self._sampling_size is None:
#             self._sampling_size = preds.shape[0]
#         self._targets.append(targets.detach().clone())
#         self._accumulated_preds.append(preds.detach().clone())
#         self._variances.append(preds.var(dim=0))

#     def reset(self):
#         self._sampling_size = None
#         self._accumulated_preds = []
#         self._targets = []
#         self._variances = []

#     def probs_mean(self) -> torch.Tensor:
#         return torch.stack(self._accumulated_preds).mean(dim=0)

#     def probs_variance(self) -> torch.Tensor:
#         return torch.stack(self._variances).var(dim=0)

# class Entropy(torchmetrics.Metric):
#     higher_is_better: bool = False
#     def __init__(self, eps: float = 1e-12, num_classes: Optional[int] = None, 
#                   probs_accumulator: Optional[ProbsAccumulator] = None) -> None:
#         self.eps = eps
#         self.num_classes = num_classes
#         self.probs_accumulator = probs_accumulator if probs_accumulator is not None else ProbsAccumulator(main_feeder=self)
#         super().__init__()

#     def update(self,
#                preds: torch.Tensor,
#                targets: torch.Tensor) -> None:
#         self.probs_accumulator.update(preds, targets, self)

#     def compute(self) -> torch.Tensor:
#         return predictive_shannon_entropy(self.probs_accumulator.mean_probs, self.eps)

# def predictive_shannon_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """
#     :param probs: of shape (*, C), with * is 0 or N, the number of sampled for which the entropy should be computed
#     :param eps:
#     :return:
#     """
#     assert_contains([1, 2], len(probs.shape), f"Expected shape (C) or (N, C), but got shape {probs.shape}")
#     assert_not_nan(probs)
#     should_reshape: bool = len(probs.shape) == 1
#     if should_reshape:
#         probs = probs[None, :]
#     assert_tensor_close(torch.ones((probs.size(0))).to(probs.device), probs.sum(dim=1))

#     Logger.get_current_logger().debug(f"Computing shannon entropy for probs of shape {probs.shape}.")
#     entropy = torch.tensor(scipy.stats.entropy(probs.detach().cpu().numpy(), base=2, axis=1), device=probs.device)

#     Logger.get_current_logger().debug(f"Computed shannon entropy with shape {entropy.shape}")
#     assert_not_nan(entropy)

#     assert_equals([probs.size(0)], list(entropy.size()))
#     if should_reshape:
#         probs = torch.squeeze(probs)
#         entropy = torch.squeeze(entropy)
#     return entropy

# class EnsembleFloatMetric(torchmetrics.Metric, ABC):
#     _probs_accumulator: ProbsAccumulator

#     def __init__(self, probs_accumulator: Optional[ProbsAccumulator] = None) -> None:
#         if probs_accumulator is None:
#             self.probs_accumulator = ProbsAccumulator(warn_mode=True, main_feeder=self)
#         else:
#             self.probs_accumulator = probs_accumulator
#         self.reset()
#         super().__init__()

#     def feed_probs(self,
#                    preds: torch.Tensor,
#                    target: Optional[torch.Tensor] = None) -> None:
#         feed_params: Dict[Text, Any] = {key: value for (key, value) in locals().items() if key != "self"}
#         self.probs_accumulator.feed_probs(preds, gt_labels=target, samples=samples, sampling_index=sampling_index, batch_index=batch_index, feeder=self)
#         self.apply_feed_hooks(feed_params)

#     def get_metric_value(self, intermediate: bool = False) -> float:
#         if not intermediate:
#             self._metric_computed = True

#         return self.get_deterministic_metric_value(self.probs_accumulator.mean_probs, self.probs_accumulator.gt_labels)

#     @abstractmethod
#     def get_deterministic_metric_value(self, probs: torch.Tensor, gt_labels: torch.Tensor) -> float:
#         ...

#     def get_metric_values_per_samples(self) -> torch.Tensor:
#         values_per_samples = self.get_deterministic_metric_values_per_samples(self.probs_accumulator.mean_probs, self.probs_accumulator.gt_labels)
#         tmp_output = self.apply_get_per_samples_hooks(values_per_samples)
#         if tmp_output is not None:
#             values_per_samples = tmp_output

#         return values_per_samples

#     @abstractmethod
#     def get_deterministic_metric_values_per_samples(self, probs: torch.Tensor, gt_labels: torch.Tensor) -> torch.Tensor:
#         ...

#     def reset(self) -> None:
#         self.probs_accumulator.reset()
#         self._metric_computed = False
        