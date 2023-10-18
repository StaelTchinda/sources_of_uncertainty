from typing import List, Union
import torch
import laplace
import laplace.utils.matrix

from typing import Tuple

from util import verification
from network import lightning as lightning
from util.tensor.kron import KronBlockDecomposed

def compute_laplace_eigenvalues(laplace_curv: laplace.ParametricLaplace) -> Union[torch.Tensor, KronBlockDecomposed]:
    if isinstance(laplace_curv, laplace.FullLaplace):
        eigenvalues = torch.linalg.eigvalsh(laplace_curv.posterior_covariance)
    elif isinstance(laplace_curv, laplace.KronLaplace):
        posterior_precision: laplace.utils.matrix.KronDecomposed = laplace_curv.posterior_precision
        verification.check_is_instance(posterior_precision, laplace.utils.matrix.KronDecomposed)
        eigenvalues = posterior_precision.eigenvalues
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")
    
    return eigenvalues


def compute_laplace_eigendecomp(laplace_curv: laplace.ParametricLaplace) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[KronBlockDecomposed, KronBlockDecomposed]]:
    if isinstance(laplace_curv, laplace.FullLaplace):
        eigenvalues, eigenvectors = torch.linalg.eigh(laplace_curv.posterior_covariance)
    elif isinstance(laplace_curv, laplace.KronLaplace):
        posterior_precision: laplace.utils.matrix.KronDecomposed = laplace_curv.posterior_precision
        verification.check_is_instance(posterior_precision, laplace.utils.matrix.KronDecomposed)
        eigenvalues = posterior_precision.eigenvalues
        eigenvectors = posterior_precision.eigenvectors
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")
    
    return eigenvalues, eigenvectors
