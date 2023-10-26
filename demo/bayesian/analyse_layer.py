
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
import matplotlib.pyplot as plt

def add_src_to_path(demo_dir_path: Optional[Path] = None):
    import sys, pathlib
    if demo_dir_path is None:
        demo_dir_path = pathlib.Path(__file__).parent.resolve()
    src_dir_path: Path = demo_dir_path.parent.parent / "src"

    sys.path.append(str(src_dir_path))


add_src_to_path()

from util import assertion, checkpoint, verification, data as data_utils
from util import lightning as lightning_util, plot as plot_util
from network.bayesian import mc_dropout as bayesian_mc_dropout
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

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='mnist', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='lenet5', required=False)

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=True)

    return parser.parse_args()


def histogram_variance(variance: torch.Tensor, title: Optional[Text] = None, channel_wise: bool = True):
    if len(variance.shape) >=2 and channel_wise:
        # Reshape the variance to a 2x2 matrix, where the second dimension is the number of channels
        data = variance.reshape(-1, variance.shape[0])
        label = [f"channel {i}" for i in range(variance.shape[0])]
    else:
        data = variance.flatten()
        label = None
    data = data.detach().cpu().numpy()

    fig, ax = plt.subplots()

    # We can set the number of bins with the *bins* keyword argument.
    if label is not None:
        ax.hist(data, label=label, stacked=True)
        ax.legend()
    else:
        ax.hist(data)

    if title is not None:
        ax.set_title(title)
    return fig


def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "bayesian" / "analyse" / "layer"
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
        pl_module = pl_module.__class__.load_from_checkpoint(str(best_checkpoint_path), model=model)
        pl_module.eval()
    else:
        pretrained_path = checkpoint.find_pretrained(model_checkpoints_path)
        if pretrained_path is not None:
            utils.verbose_and_log(f"Loading pretrained model from {pretrained_path}", args.verbose, args.log)
            checkpoint.load_model(model, file_name=pretrained_path.stem, 
                                path_args={'save_path': pretrained_path.parent, 'file_ext': pretrained_path.suffix[1:]})
        else:   
            raise ValueError("No checkpoint or pretrained model found")


    # Set the Monte Carlo dropout
    dropout_hook: bayesian_mc_dropout.mc_dropout.DropoutHook
    if hasattr(model, "dropout_hook"):
        dropout_hook = model.dropout_hook
    else:
        dropout_hook = bayesian_mc_dropout.mc_dropout.DropoutHook(model)

    # Goal 1: anaylse the evolution of the variance over the layers
    # Approach:
    # - add a/multiple hook(s) to the model, which 
    #    - computes the variance of the output of each layer
    #    - saves it
    #    - logs it
    # - run the model on the validation set
    variance_callback = SaveLayerVarianceCallback()

    mc_dropout_pl_module = config.bayesian.mc_dropout.lightning.get_default_lightning_mc_dropout_module(model_mode, dropout_hook)
    mc_dropout_trainer = config.bayesian.laplace.lightning.get_default_lightning_laplace_trainer(
        model_mode, 
        {
            "default_root_dir": log_path,
            "callbacks": [variance_callback]
        }
    )

    mc_dropout_trainer.validate(mc_dropout_pl_module, data_module)
    if args.verbose:
        print("Finished validation")

    if isinstance(mc_dropout_trainer.logger, pl_tensorboard.TensorBoardLogger):
        for module_name, variance in variance_callback.named_variances().items():
            # IDEA: use the native function of tensorboard to log the histogram: SummaryWriter.add_histogram
            mc_dropout_trainer.logger.experiment.add_figure(f"analyse_layer/{module_name}", histogram_variance(variance, title=module_name, channel_wise=False))
            mc_dropout_trainer.logger.experiment.add_figure(f"analyse_layer/channel/{module_name}", histogram_variance(variance, title=module_name, channel_wise=True))

    else:
        warnings.warn("No TensorBoardLogger found, variances cannot be logged")

utils.register_cleanup()

if __name__ == '__main__':
    utils.catch_and_print(main)
