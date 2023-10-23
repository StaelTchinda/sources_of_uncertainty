from typing import Dict, Text, Any
import laplace



from config import metrics as metrics_config
from network.lightning.laplace import LaplaceModule
from network.lightning import laplace as bayesian_laplace

def get_default_lightning_laplace_trainer_params() -> Dict[Text, Any]:
    return {
        "devices": 1,
        "deterministic": True, # To allow reproducibility
        "enable_progress_bar": True,  
        "inference_mode": False # To allow gradient computation in validation mode # See https://github.com/Lightning-AI/lightning/issues/18222
    }


def get_default_laplace_lightning_module_params() -> Dict[Text, Any]:
    return {
        "prediction_mode": "bayesian",
        "pred_type": "glm",
        "n_samples": 50,
        "val_metrics": metrics_config.get_default_ensemble_val_metrics(num_classes=10)
    }

def get_default_lightning_laplace_module(laplace: laplace.ParametricLaplace) -> bayesian_laplace.LaplaceModule:
    params = get_default_laplace_lightning_module_params()
    return LaplaceModule(laplace, **params)