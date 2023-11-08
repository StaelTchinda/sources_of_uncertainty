from typing import Dict, Text,Any

import laplace
from laplace import curvature as laplace_curv


def get_default_laplace_name(model_mode) -> Text:
    if model_mode == "resnet20":
        return "resnet20_laplace_kron"
    elif model_mode == "resnet32":
        return "resnet32_laplace_kron"
    elif model_mode == "resnet44":
        return "resnet44_laplace_kron"
    else:
        raise NotImplementedError(f"Model mode {model_mode} not implemented")

def get_default_laplace_params() -> Dict[Text, Any]:
    return {
        "likelihood": "classification",
        "subset_of_weights": "all",
        "hessian_structure": "kron",
        "backend": laplace_curv.AsdlGGN,
        "ignore_batchnorm": False,
        "backend_kwargs": {
            "ignore_batchnorm": False,
        }
    }

def get_default_prior_optimization_params() -> Dict[Text, Any]:
    return {
        "method": "marglik",
        "pred_type": "glm",
        "link_approx": "probit",
        "n_samples": 100,
    }