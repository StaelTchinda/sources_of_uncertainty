
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Text
import logging
import pytorch_lightning as pl
import torch
from torch.utils import data as torch_data
import torch.distributions as dists
from netcal import metrics as netcal

def add_src_to_path(demo_dir_path: Optional[Path] = None):
    import sys, pathlib
    if demo_dir_path is None:
        demo_dir_path = pathlib.Path(__file__).parent.resolve()
    src_dir_path: Path = demo_dir_path.parent.parent / "src"

    sys.path.append(str(src_dir_path))


add_src_to_path()

from util import checkpoint, data as data_utils
from util import lightning as lightning_util
from config.laplace import get_default_laplace_params
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils
import config


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

    return parser.parse_args()

def main():
    args: argparse.Namespace = parse_args()

    model_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    laplace_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "laplace" / "eval"
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

    best_checkpoint_path = checkpoint.find_best_checkpoint(model_log_path)
    if best_checkpoint_path is not None:
        pl_module.load_from_checkpoint(str(best_checkpoint_path), model=model)
        pl_module.eval()
    else:
        raise ValueError("No checkpoint found")

    # Initialize the LaPlace approximation
    laplace_filename = config.laplace.get_default_laplace_name(model_mode)
    laplace = checkpoint.load_object(laplace_filename, path_args={"save_path": log_path / 'laplace'}, library='dill')
    laplace_params = get_default_laplace_params(model_mode)
    if laplace is None:
        data_module.setup("validate")
        laplace = bayesian_laplace.compute_laplace_for_model(model, data_module.train_dataloader(), data_module.val_dataloader(), laplace_params)
        checkpoint.save_object(laplace, laplace_filename, save_path=log_path / 'laplace', library='dill')
    
    laplace_pl_module = config.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace) 
    laplace_trainer = config.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": laplace_log_path})
    laplace_trainer.validate(laplace_pl_module, data_module)

    # Evalauate the LaPlace approximation
    data_module.setup("validate")
    utils.evaluate_model(laplace, data_module.val_dataloader(), "LaPlace")
    
if __name__ == '__main__':
    main()
        

