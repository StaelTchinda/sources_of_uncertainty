
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Text, Union
import logging
import pytorch_lightning as pl
import torch
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

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='mnist', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='lenet5', required=False)

    parser.add_argument('--joint_data', help='specify which joint dataset to use', type=str, choices=config.mode.AVAILABLE_JOINT_DATASETS, default=None, required=False)

    parser.add_argument('--stage', help='specify which stage to use', type=str, choices=['val', 'test'], default='test', required=False)

    parser.add_argument('--bayesian', help='specify which bayesian method to use', type=str, choices=config.mode.AVAILABLE_BAYESIAN_MODES, default='mc_dropout', required=False)

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    laplace_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "bayesian" / "laplace"
    if args.joint_data is None:
        log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "bayesian" / 'visualisation'
    else:
        log_path: Path  = config.path.CHECKPOINT_PATH / f"{args.joint_data}" / f"{args.model}" / "bayesian" / 'visualisation'
    log_foldername: Text = f"run {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    if args.log:
        basic_config_params = config.log.get_log_basic_config(filename=log_path / f'{log_foldername}.log')
        logging.basicConfig(**basic_config_params)
        utils.verbose_and_log(f"Logging enabled: {log_foldername}", args.verbose, args.log)
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
                    
        utils.verbose_and_log(f"Model created: {pl_module}", args.verbose, args.log)
        
        # Get the dropout hook
        dropout_hook: bayesian_mc_dropout.DropoutHook
        if hasattr(model, "dropout_hook"):
            dropout_hook = getattr(model, 'dropout_hook')
        else:
            dropout_hook = bayesian_mc_dropout.DropoutHook(model)

        bayesian_module = config.bayesian.mc_dropout.lightning.get_default_lightning_mc_dropout_module(model_mode, dropout_hook)
    else:
        raise ValueError(f"Invalid bayesian method {args.bayesian}")

    bayesian_module.save_hyperparameters(args)


    # Initialize the trainer
    callback_containers = config.bayesian.laplace.eval.get_callback_containers(model_mode)
    additional_params = {
        "default_root_dir": log_path,
        "callbacks": [
            callback for callback_container in callback_containers.values() for callback in callback_container.callbacks
        ],
    }
    bayesian_trainer = config.bayesian.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, additional_params) 

    if args.stage == 'val':
        bayesian_trainer.validate(bayesian_module, data_module)
    elif args.stage == 'test':
        bayesian_trainer.test(bayesian_module, data_module)
    else:
        raise ValueError(f"Invalid stage {args.stage}")
    
    if args.data in ["mnist", "cifar10"]:
        for (container_name, callback_container) in callback_containers.items():
            bayesian_module.logger.experiment.add_figure(container_name, callback_container.plot(), global_step=0)


# Register the cleanup function to be called on exit
utils.register_cleanup()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Handle exceptions gracefully, log errors, etc.
        print("An error occurred:", str(e))
        # Print stacktrace
        import traceback
        traceback.print_exc()        

