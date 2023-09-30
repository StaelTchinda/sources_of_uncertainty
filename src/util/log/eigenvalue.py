from typing import Optional, Tuple, Text

from typing import List, Optional, Text
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import laplace
import laplace.utils.matrix

import numpy as np
from typing import Tuple

from network import lightning as lightning
from util.laplace import compute_laplace_eigenvalues
from util.network import compute_model_decomposition
from util.plot import heatmap_eigenvalues

from util.tensor.eigenvalue import ReduceMode, NetworkGranularity

def log_laplace_eigenvalues(laplace_curv: laplace.ParametricLaplace, logger: TensorBoardLogger, params: List[Tuple[NetworkGranularity, Optional[ReduceMode]]]):
    weight_eigenvalues = compute_laplace_eigenvalues(laplace_curv)
    model_decompostion = compute_model_decomposition(laplace_curv.model)
        
    if isinstance(laplace_curv, laplace.FullLaplace):
        prefix: Text = "laplace_eig/full"
    elif isinstance(laplace_curv, laplace.KronLaplace):
        prefix: Text = "laplace_eig/kron"
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")

    for (x_axis, reduce_mode) in params:
        for (layer_name, fig) in heatmap_eigenvalues(weight_eigenvalues, x_axis=x_axis, reduce_mode=reduce_mode, model_decomposition=model_decompostion).items():
            fig_prefix = f"{prefix}/{x_axis}/{reduce_mode}"
            fig_name = f"{fig_prefix}/{layer_name}" if layer_name != '' else fig_prefix
            logger.experiment.add_figure(fig_name, fig)


