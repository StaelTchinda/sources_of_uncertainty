
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

from util import assertion, config, checkpoint
from util import lightning as lightning_util
from data.dataset import utils as data_utils 
from util.config import laplace as laplace_config, network as network_config
from network.bayesian import laplace as bayesian_laplace
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

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=True)

    parser.add_argument('--eval', help='specify if the model should be evaluated', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    parser.set_defaults(eval=True)

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
    train_dataloader, val_dataloader, test_dataloader = config.dataset.get_default_dataloaders(data_mode)
    utils.verbose_and_log(f"Datasets initialized: \n{data_utils.verbose_dataloaders(train_dataloader, val_dataloader, test_dataloader)}", args.verbose, args.log)

    # Initialize the model
    model = config.network.get_default_model(model_mode)
    utils.verbose_and_log(f"Model created: {model}", args.verbose, args.log)

    pl_module: pl.LightningModule = network_config.lightning.get_default_lightning_module(model_mode, model)

    # Log the hyperparameters
    pl_module.save_hyperparameters({
        **args.__dict__, 
        "dataset_params": config.dataset.get_default_datasets_params(data_mode),
        "dataloaders_params": config.dataset.get_default_dataloaders_params(data_mode), 
        "model_params": config.network.get_default_model_params(model_mode),
        "trainer_params": config.network.lightning.get_default_lightning_trainer(model_mode),})

    # Eventually load the model from a checkpoint
    best_checkpoint_path: Optional[Path] = None
    if args.checkpoint:
        best_checkpoint_path = checkpoint.find_best_checkpoint(log_path)
        if best_checkpoint_path is not None:
            pl_module.load_from_checkpoint(str(best_checkpoint_path), model=model)
            # pl_module.eval()

    # Eventually train the model
    trainer = network_config.lightning.get_default_lightning_trainer(model_mode, {"default_root_dir": log_path})
    if best_checkpoint_path is None:
        trainer.fit(pl_module, train_dataloader, val_dataloader)

    # Evaluate the model
    if args.eval:
        faster_train_dataloader = torch_data.DataLoader(train_dataloader.dataset, batch_size=val_dataloader.batch_size, shuffle=False, num_workers=train_dataloader.num_workers)
        trainer.validate(pl_module, faster_train_dataloader)
        pl_module.eval()
        utils.evaluate_model(pl_module, faster_train_dataloader, "MAP_on_train")
        trainer.validate(pl_module, val_dataloader)
        pl_module.eval()
        utils.evaluate_model(pl_module, val_dataloader, "MAP_on_val")
    
if __name__ == '__main__':
    main()
        

