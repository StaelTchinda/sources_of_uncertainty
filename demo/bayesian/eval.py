
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Text, Union
import logging
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils import data as torch_data
import torch.distributions as dists
from netcal import metrics as netcal
import laplace


def add_src_to_path(demo_dir_path: Optional[Path] = None):
    import sys, pathlib
    if demo_dir_path is None:
        demo_dir_path = pathlib.Path(__file__).parent.resolve()
    src_dir_path: Path = demo_dir_path.parent.parent / "src"

    sys.path.append(str(src_dir_path))


add_src_to_path()

from util import assertion, checkpoint
from util import lightning as lightning_util, data as data_utils
import config
from network.bayesian import laplace as bayesian_laplace, mc_dropout as bayesian_mc_dropout
from network import lightning as lightning
from util import utils
from test import bottleneck

from network.pruning import pruner as pruning_wrapper, util as pruning_util
from callbacks import keep_sample


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
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='vgg11', required=False)

    parser.add_argument('--stage', help='specify which stage to use', type=str, choices=['val', 'test'], default='val', required=False)

    parser.add_argument('--bayesian', help='specify which bayesian method to use', type=str, choices=config.mode.AVAILABLE_BAYESIAN_MODES, default='mc_dropout', required=False)

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    bayesian_log_path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "bayesian"
    laplace_log_path: Path = bayesian_log_path / "laplace"
    log_path: Path = bayesian_log_path / 'eval'
    log_foldername: Text = f"run {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    if args.log:
        basic_config_params = config.log.get_log_basic_config(filename=log_path / f'{log_foldername}.log')
        logging.basicConfig(**basic_config_params)
        utils.verbose_and_log(f"Logging enabled: {log_foldername}", args.verbose, args.log)
    else:
        logging.disable(logging.CRITICAL)

    utils.verbose_and_log(f"Running with arguments: {args}", args.verbose, args.log)

    data_mode: config.mode.DataMode = args.data
    model_mode: config.mode.ModelMode = args.model

    # Initialize the dataloaders
    data_module = config.data.lightning.get_default_datamodule(data_mode)
    utils.verbose_and_log(f"Datamodule initialized: \n{data_utils.verbose_datamodule(data_module)}", args.verbose, args.log)

    # Initialize the bayesian approximation
    if args.bayesian == 'laplace':
        # Initialize the laplace approximation
        laplace_filename = config.bayesian.laplace.get_default_laplace_name(model_mode)
        utils.verbose_and_log(f"Loading LaPlace approximation with name {laplace_filename} from {laplace_log_path}", args.verbose, args.log)
        laplace_curv: laplace.ParametricLaplace = checkpoint.load_object(laplace_filename, path_args={"save_path": laplace_log_path}, library='dill')
        if laplace_curv is None:
            raise ValueError("No laplace approximation found")
        utils.verbose_and_log(f"Laplace loaded to model: {laplace_curv.model}", args.verbose, args.log)

        bayesian_module = config.bayesian.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv) 
    elif args.bayesian == 'mc_dropout':
        # Initialize the model
        model = config.network.get_default_model(model_mode)
        utils.verbose_and_log(f"Model created: {model}", args.verbose, args.log)
        
        # Load the best checkpoint
        best_checkpoint_path = checkpoint.find_best_checkpoint(model_checkpoints_path)
        if best_checkpoint_path is not None:
            pl_module: pl.LightningModule = config.network.lightning.get_default_lightning_module(model_mode, model)
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
        
        # Get the dropout hook
        dropout_hook: bayesian_mc_dropout.DropoutHook
        if hasattr(model, "dropout_hook"):
            dropout_hook = model.dropout_hook
        else:
            dropout_hook = bayesian_mc_dropout.DropoutHook(model)

        bayesian_module = config.bayesian.mc_dropout.lightning.get_default_lightning_mc_dropout_module(model_mode, dropout_hook)
    else:
        raise ValueError(f"Invalid bayesian method {args.bayesian}")

    bayesian_module.save_hyperparameters(args)


    # Initialize the trainer
    additional_params = {
        "default_root_dir": log_path
    }
    bayesian_trainer = config.bayesian.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, additional_params) 

    bayesian_module.eval()
    model.eval()
    if args.stage == 'val':
        bayesian_trainer.validate(bayesian_module, data_module)
        # utils.evaluate_model(nn.Sequential(model, nn.Softmax(dim=-1)), data_module.val_dataloader(), 'MAP')
    elif args.stage == 'test':
        bayesian_trainer.test(bayesian_module, data_module)
        # utils.evaluate_model(bayesian_module, data_module.test_dataloader(), 'MAP')
    else:
        raise ValueError(f"Invalid stage {args.stage}")


# Register the cleanup function to be called on exit
utils.register_cleanup()

if __name__ == '__main__':
    utils.catch_and_print(lambda: utils.run_with_timeout(main, timeout=40*60))
