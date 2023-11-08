
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Text
import logging
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils import data as torch_data
import torch.distributions as dists
from netcal import metrics as netcal

import util

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
from network import lightning as lightning
from util import utils


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

    return parser.parse_args()

def main():
    args: argparse.Namespace = parse_args()

    model_checkpoints_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    if args.joint_data is None:
        log_path: Path = model_checkpoints_path / "eval"
    else:
        log_path: Path = config.path.CHECKPOINT_PATH / f"{args.joint_data}" / f"{args.model}" / "model" / "eval"
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
    verbose_stages = config.bayesian.laplace.data.lightning.get_default_verbose_stages(data_mode)
    utils.verbose_and_log(f"Datamodule initialized: \n{data_utils.verbose_datamodule(data_module, stages=verbose_stages)}", args.verbose, args.log)
    
    # Load the best checkpoint
    best_checkpoint_model, best_checkpoint_path = config.network.checkpoint.get_best_checkpoint(model_mode, model_checkpoints_path, with_path=True)
    if best_checkpoint_model is not None:
        utils.verbose_and_log(f"Loaded best checkpoint model from {best_checkpoint_path}", args.verbose, args.log)
        pl_module = best_checkpoint_model
        pl_module.eval()
    else:
        pretrained_model, pretrained_path = config.network.checkpoint.get_pretrained(model_mode, model_checkpoints_path, with_path=True)
        if pretrained_model is not None:
            utils.verbose_and_log(f"Loaded pretrained model from {pretrained_path}", args.verbose, args.log)
            pl_module: pl.LightningModule = config.network.lightning.get_default_lightning_module(model_mode, pretrained_model)
            pl_module.eval()
        else:   
            raise ValueError("No checkpoint or pretrained model found")
                
    utils.verbose_and_log(f"Model created: {pl_module}", args.verbose, args.log)


    # Initialize the trainer
    additional_params = {"default_root_dir": log_path}
    trainer = config.network.lightning.get_default_lightning_trainer(model_mode, additional_params)

    # Evaluate the model
    if args.stage=='val':
        trainer.validate(pl_module, data_module)
        pl_module.eval()
        utils.evaluate_model(pl_module, data_module.val_dataloader(), "MAP")
    elif args.stage=='test':
        trainer.test(pl_module, data_module)
        pl_module.eval()
        utils.evaluate_model(pl_module, data_module.test_dataloader(), "MAP")
    
# Register the cleanup function to be called on exit
utils.register_cleanup()

if __name__ == '__main__':
    utils.catch_and_print(main)
