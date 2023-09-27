from typing import Dict, Text,Any

import laplace
from laplace import curvature as laplace_curv


def get_default_laplace_name() -> Text:
    # return "fc_laplace_full"
    return "fc_laplace_kron"

def get_default_laplace_params() -> Dict[Text, Any]:
    return {
        "likelihood": "classification",
        "subset_of_weights": "all",
        # "hessian_structure": "full",
        "hessian_structure": "kron",
        "backend": laplace_curv.AsdlGGN
    }

def get_default_prior_optimization_params() -> Dict[Text, Any]:
    return {
        "method": "marglik",
        "pred_type": "glm",
        "link_approx": "probit",
        "n_samples": 1000,
    }