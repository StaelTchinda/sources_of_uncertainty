
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

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='cifar10', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='resnet20', required=False)

    parser.add_argument('--joint_data', help='specify which joint dataset to use', type=str, choices=config.mode.AVAILABLE_JOINT_DATASETS, default=None, required=False)

    parser.add_argument('--stage', help='specify which stage to use', type=str, choices=['val', 'test'], default='val', required=False)

    return parser.parse_args()


def histogram_variance(variance: torch.Tensor, title: Optional[Text] = None, channel_wise: bool = True):
    if len(variance.shape) >=2 and channel_wise:
        # Reshape the variance to a 2x2 matrix, where the second dimension is the number of channels
        variance = variance.reshape(-1, variance.shape[0])
        label = [f"channel {i}" for i in range(variance.shape[0])]
    else:
        variance = variance.flatten()
        label = None
    variance = variance.detach().cpu().numpy()

    fig, ax = plt.subplots()

    # We can set the number of bins with the *bins* keyword argument.
    if label is not None:
        ax.hist(variance, label=label, stacked=True)
        ax.legend()
    else:
        ax.hist(variance)

    if title is not None:
        ax.set_title(title)
    return fig


def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    if args.joint_data is None:
        log_path: Path = model_checkpoints_path / "bayesian" / "analyse" / "layer"
    else:
        log_path: Path = config.path.CHECKPOINT_PATH / f"{args.joint_data}" / f"{args.model}" / "bayesian" / "analyse" / "layer"
    log_filename: Text = f"run {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    if args.log:
        basic_config_params = config.log.get_log_basic_config(filename=log_path / f'{log_filename}.log')
        logging.basicConfig(**basic_config_params)
        utils.verbose_and_log(f"Logging enabled: {log_filename}", args.verbose, args.log)
    else:
        logging.disable(logging.CRITICAL)

    data_mode: config.mode.DataMode = args.data
    joint_data_mode: Optional[config.mode.JointDataMode] = args.joint_data
    model_mode: config.mode.ModelMode = args.model

    # Initialize the dataloaders
    if joint_data_mode is None:
        data_module = config.data.lightning.get_default_datamodule(data_mode)
    else:
        data_module = config.data.lightning.get_default_joint_datamodule(joint_data_mode)
    utils.verbose_and_log(f"Datamodule initialized: \n{data_utils.verbose_datamodule(data_module)}", args.verbose, args.log)

    # Load the best checkpoint
    best_checkpoint_module, best_checkpoint_path = config.network.checkpoint.get_best_checkpoint(model_mode, model_checkpoints_path, with_path=True)
    if best_checkpoint_module is not None:
        utils.verbose_and_log(f"Loaded best checkpoint model from {best_checkpoint_path}", args.verbose, args.log)
        pl_module = best_checkpoint_module
        model = pl_module.model
        pl_module.eval()
    else:
        pretrained_model, pretrained_path = config.network.checkpoint.get_pretrained(model_mode, model_checkpoints_path, with_path=True)
        if pretrained_model is not None:
            utils.verbose_and_log(f"Loaded pretrained model from {pretrained_path}", args.verbose, args.log)
            model = pretrained_model
            pl_module: pl.LightningModule = config.network.lightning.get_default_lightning_module(model_mode, pretrained_model)
            pl_module.eval()
        else:   
            raise ValueError("No checkpoint or pretrained model found")
            

    # Set the Monte Carlo dropout
    dropout_hook: bayesian_mc_dropout.DropoutHook
    if hasattr(model, "dropout_hook"):
        dropout_hook = getattr(model, 'dropout_hook')
    else:
        raise ValueError("No dropout hook found")

    # Goal 1: anaylse the evolution of the variance over the layers
    # Approach:
    # - add a/multiple hook(s) to the model, which 
    #    - computes the variance of the output of each layer
    #    - saves it
    #    - logs it
    # - run the model on the validation set
    
    mc_dropout_pl_module = config.bayesian.mc_dropout.lightning.get_default_lightning_mc_dropout_module(model_mode, dropout_hook)
    variance_callback = SaveLayerVarianceCallback(sampling_size=mc_dropout_pl_module._n_samples)
    mc_dropout_trainer = config.bayesian.laplace.lightning.get_default_lightning_laplace_trainer(
        model_mode, 
        {
            "default_root_dir": log_path,
            "callbacks": [variance_callback]
        }
    )

    if args.stage == 'val':
        mc_dropout_trainer.validate(mc_dropout_pl_module, data_module)
    elif args.stage == 'test':
        mc_dropout_trainer.test(mc_dropout_pl_module, data_module)
    else:
        raise ValueError(f"Invalid stage {args.stage}")
    utils.verbose_and_log(f"Finished forward passing.", args.verbose, args.log)

    if isinstance(mc_dropout_trainer.logger, pl_tensorboard.TensorBoardLogger):
        for module_name, variance in variance_callback.named_variances():
            utils.verbose_and_log(f"Logging variance of module {module_name} of shape {variance.shape}", args.verbose, args.log)
            # IDEA: use the native function of tensorboard to log the histogram: SummaryWriter.add_histogram
            mc_dropout_trainer.logger.experiment.add_figure(f"analyse_layer/{args.stage}/{module_name}", histogram_variance(variance, title=module_name, channel_wise=False))
            mc_dropout_trainer.logger.experiment.add_figure(f"analyse_layer/{args.stage}/channel/{module_name}", histogram_variance(variance, title=module_name, channel_wise=True))
    else:
        warnings.warn("No TensorBoardLogger found, variances cannot be logged")

utils.register_cleanup()

if __name__ == '__main__':
    utils.catch_and_print(main)
