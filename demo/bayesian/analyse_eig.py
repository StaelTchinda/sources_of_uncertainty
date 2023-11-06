
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Text, Union
import logging
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.utils import data as torch_data
import torch.distributions as dists
from netcal import metrics as netcal
import laplace
import laplace.utils.matrix

def add_src_to_path(demo_dir_path: Optional[Path] = None):
    import sys, pathlib
    if demo_dir_path is None:
        demo_dir_path = pathlib.Path(__file__).parent.resolve()
    src_dir_path: Path = demo_dir_path.parent.parent / "src"

    sys.path.append(str(src_dir_path))


add_src_to_path()

from util.log.eigenvalue import log_laplace_eigenvalues
from util import assertion, checkpoint, verification, data as data_utils
from util import lightning as lightning_util, plot as plot_util
import config
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils
import metrics



# Take script arguments for the data mode and the model mode
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Say hello')

    parser.add_argument('--log', help='specify if the process should be logged', action='store_true')
    parser.add_argument('--no-log', dest='log', action='store_false')
    parser.set_defaults(log=False)

    parser.add_argument('--verbose', help='specify if the process should be verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='cifar10', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='lenet5', required=False)

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=True)

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    bayesian_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "bayesian"
    laplace_checkpoints_path: Path = bayesian_checkpoints_path /  "laplace"
    log_path: Path = bayesian_checkpoints_path / "analyse" / "eig"
    log_filename: Text = f"run {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    if args.log:
        basic_config_params = config.log.get_log_basic_config(filename=log_path / f'{log_filename}.log')
        logging.basicConfig(**basic_config_params)
        utils.verbose_and_log(f"Logging enabled: {log_filename}", args.verbose, args.log)
    else:
        logging.disable(logging.CRITICAL)

    model_mode: config.mode.ModelMode = args.model

    # Load the LaPlace approximation
    laplace_filename = config.bayesian.laplace.get_default_laplace_name(model_mode)
    laplace_curv: laplace.ParametricLaplace = None
    if args.checkpoint is True:
        laplace_curv = checkpoint.load_object(laplace_filename, path_args={"save_path": laplace_checkpoints_path}, library='dill')
        utils.verbose_and_log(f"LaPlace approximation loaded from {laplace_filename}", args.verbose, args.log)
    if laplace_curv is None:
        raise ValueError("No LaPlace approximation found")

    # Initialize the LaPlace module
    laplace_pl_module = config.bayesian.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
    laplace_trainer = config.bayesian.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": log_path})

    # Observe the eigenvalues of the Hessian
    utils.verbose_and_log(f"Observe the eigenvalues of the Hessian", args.verbose, args.log)
    log_laplace_eigenvalues(laplace_curv, laplace_trainer.logger)
    
# Register the cleanup function to be called on exit
utils.register_cleanup()


# Since the eigenvalues of the hessian are already sorted, a heatmap is not very valuable, because it will always be from lowest to highest (from left to right). A histogram would make more sense to analyse how much distributed the eigenvalues are. Labeling after layers or even channels is not possible.
# This is however possible for the PCA loadings, as they are linked to the original weights. 

if __name__ == '__main__':
    utils.catch_and_print(main)
