from typing import Callable, Iterable

import torch

BayesianModuleLike = Callable[[torch.Tensor], Iterable[torch.Tensor]]

