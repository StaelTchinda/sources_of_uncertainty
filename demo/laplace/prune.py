
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

from util import assertion, checkpoint, verification
from util import lightning as lightning_util, plot as plot_util, data as data_utils
# from config import laplace as laplace_config, network as network_config, metrics as metrics_config, data as data_config
import config
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning, pruning
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

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='mnist', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='lenet5', required=False)

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=True)

    parser.add_argument('--pruning', help='specify which pruning to use', choices=pruning.wrapper.AVAILABLE_PRUNING_STRATEGIES, default='random', required=False)

    parser.add_argument('--pruning_sparsities', help='specify which sparsity we want the model to have', nargs='*', default=[0.1, 0.3, 0.5, 0.7, 0.9], required=False)

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}"
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
    best_checkpoint_path = checkpoint.find_best_checkpoint(log_path)
    if best_checkpoint_path is not None:
        pl_module.load_from_checkpoint(str(best_checkpoint_path), model=model)
        pl_module.eval()
    else:
        raise ValueError("No checkpoint found")

    # Load the LaPlace approximation
    laplace_filename = config.laplace.get_default_laplace_name(model_mode)
    laplace_curv: laplace.ParametricLaplace = None
    if args.checkpoint is True:
        laplace_curv = checkpoint.load_object(laplace_filename, path_args={"save_path": log_path / 'laplace'}, library='dill')
    if laplace_curv is None:
        raise ValueError("No LaPlace approximation found")

    # Initialize the LaPlace module
    laplace_pl_module = config.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
    laplace_trainer = config.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": log_path})


    # Observe the eigenvalues of the Hessian
    pruning_module = pruning.wrapper.PruningModule(pl_module.model, 
                                                   config.metrics.get_default_deterministic_val_metrics(num_classes=3))
    pruning_amounts = utils.get_required_amounts_for_sparsities(args.pruning_sparsities)
    trainer = pl.Trainer(devices=1, deterministic=True, default_root_dir=log_path)
    for module_name in pruning_module.prunable_named_modules().keys():
        for pruning_amount in pruning_amounts:
            pruning_module.configure_pruning(module_name, args.pruning, pruning_amount)
            pruning_module.prune()
            trainer.validate(pruning_module, data_module)

if __name__ == '__main__':
    main()
        

