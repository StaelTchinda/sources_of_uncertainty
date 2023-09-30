
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Text, Union
import logging
import warnings
import pytorch_lightning as pl
from pytorch_lightning.loggers import tensorboard as pl_tensorboard
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.tensorboard.writer import SummaryWriter
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

from util import assertion, checkpoint, verification, data as data_utils
from util import lightning as lightning_util, plot as plot_util
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils
import metrics
import config
from callbacks.layer_variance import SaveLayerVarianceCallback


# Take script arguments for the data mode and the model mode
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Say hello')

    parser.add_argument('--log', help='specify if the process should be logged', action='store_true')
    parser.add_argument('--no-log', dest='log', action='store_false')
    parser.set_defaults(log=False)

    parser.add_argument('--verbose', help='specify if the process should be verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='iris', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='fc', required=False)

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=True)

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    laplace_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "laplace"
    log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "laplace" / "analyse" / "layer"
    log_filename: Text = f"run {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    if args.log:
        basic_config_params = config.log.get_log_basic_config(filename=log_path / f'{log_filename}.log')
        logging.basicConfig(**basic_config_params)
        utils.verbose_and_log(f"Logging enabled: {log_filename}", args.verbose, args.log)
    else:
        logging.disable(logging.CRITICAL)

    data_mode: config.mode.DataMode = args.data
    model_mode: config.mode.ModelMode = args.model

    # Initialize the dataloaders
    data_module = config.data.lightning.get_default_datamodule(data_mode)
    utils.verbose_and_log(f"Datamodule initialized: \n{data_utils.verbose_datamodule(data_module)}", args.verbose, args.log)


    # Initialize the model
    model = config.network.get_default_model(model_mode)
    utils.verbose_and_log(f"Model created: {model}", args.verbose, args.log)
    pl_module: pl.LightningModule = config.network.lightning.get_default_lightning_module(model_mode, model)

    # Load the best checkpoint
    best_checkpoint_path = checkpoint.find_best_checkpoint(model_checkpoints_path)
    if best_checkpoint_path is not None:
        pl_module.load_from_checkpoint(str(best_checkpoint_path), model=model)
        pl_module.eval()
    else:
        raise ValueError(f"No checkpoint found at {model_checkpoints_path}")

    # Load the LaPlace approximation
    laplace_filename = config.laplace.get_default_laplace_name(model_mode)
    laplace_curv: laplace.ParametricLaplace = None
    if args.checkpoint is True:
        laplace_curv = checkpoint.load_object(laplace_filename, path_args={"save_path": laplace_checkpoints_path}, library='dill')
    if laplace_curv is None:
        raise ValueError("No LaPlace approximation found")

    # Goal 1: anaylse the evolution of the variance over the layers
    # Approach:
    # - add a/multiple hook(s) to the model, which 
    #    - computes the variance of the output of each layer
    #    - saves it
    #    - logs it
    # - run the model on the validation set
    variance_callback = SaveLayerVarianceCallback()

    # Initialize the LaPlace module
    laplace_pl_module = config.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
    laplace_trainer = config.laplace.lightning.get_default_lightning_laplace_trainer(
        model_mode, 
        {
            "default_root_dir": log_path,
            "callbacks": [variance_callback]
        }
    )

    laplace_trainer.validate(laplace_pl_module, data_module)
    if args.verbose:
        print("Finished validation")

        print(f"Variances per layer")
        for module, variance in variance_callback.named_variances().items():
            print(f"{module} : {variance}")

    if isinstance(laplace_trainer.logger, pl_tensorboard.TensorBoardLogger):
        for module_name, variance in variance_callback.named_variances().items():
            try:
                laplace_trainer.logger.experiment.add_histogram(f"laplace_layer/{module_name}", variance.detach().cpu().numpy(), 0)
            except TypeError as e:
                from util.log.histogram import histogram_vector
                laplace_trainer.logger.experiment.add_figure(f"laplace_layer/{module_name}", histogram_vector(variance.detach().cpu().numpy()))

                laplace_trainer.logger.experiment.add_text("error", f"Could not log variances: {e}")
                warnings.warn(f"Could not log variances: {e}")
    else:
        warnings.warn("No TensorBoardLogger found, variances cannot be logged")




if __name__ == '__main__':
    main()
        

