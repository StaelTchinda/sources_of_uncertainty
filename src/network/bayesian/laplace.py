from typing import Any, Dict, Optional, Text

from laplace import Laplace, ParametricLaplace

from torch import nn
from torch.utils.data import DataLoader

from network.bayesian.util import BayesianModuleLike
from util import verification, data as data_util


def make_module_bayesian_like(model: nn.Module, laplace: ParametricLaplace, sampling_size: int = 10) -> BayesianModuleLike:
    verification.check_is(model, laplace.model)
    bayesian_model: BayesianModuleLike = lambda x: laplace.predictive_samples(x, n_samples=sampling_size)
    # bayesian_model: BayesianModuleLike = lambda x: laplace.predictive_samples(x, pred_type="nn", n_samples=sampling_size)
    return bayesian_model


def compute_laplace_for_model(torch_model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader,
                              laplace_params: Optional[Dict[Text, Any]] = None, prior_optimization_params: Optional[Dict[Text, Any]] = None,
                              verbose: bool = False) -> ParametricLaplace:
    if laplace_params is None:
        laplace_params = {}
    if prior_optimization_params is None:
        prior_optimization_params = {}
    if verbose:
        train_tqdm_params = {"desc": "Fitting Laplace", "unit": "batch"}
        train_dataloader = data_util.TqdmDataLoader(train_dataloader, train_tqdm_params)
        val_tqdm_params = {"desc": "Computing Laplace prior", "unit": "batch"}
        val_dataloader = data_util.TqdmDataLoader(val_dataloader, val_tqdm_params)
    if 'val_loader' not in prior_optimization_params:
        prior_optimization_params['val_loader'] = val_dataloader
    la: ParametricLaplace = Laplace(torch_model, **laplace_params)
    la.fit(train_dataloader)
    la.optimize_prior_precision(**prior_optimization_params)
    return la