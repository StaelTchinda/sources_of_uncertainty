
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
import laplace

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
from util.config import laplace as laplace_config, network as network_config, metrics as metrics_config
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

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='iris', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='fc', required=False)

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=True)

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
    train_dataloader, val_dataloader, test_dataloader = config.dataset.get_default_laplace_dataloaders(data_mode)
    utils.verbose_and_log(f"Datasets initialized: \n{data_utils.verbose_dataloaders(train_dataloader, val_dataloader, test_dataloader)}", args.verbose, args.log)

    # Initialize the model
    model = config.network.get_default_model(model_mode)
    utils.verbose_and_log(f"Model created: {model}", args.verbose, args.log)

    pl_module: pl.LightningModule = network_config.lightning.get_default_lightning_module(model_mode, model)
    best_checkpoint_path = checkpoint.find_best_checkpoint(log_path)
    if best_checkpoint_path is not None:
        pl_module.load_from_checkpoint(str(best_checkpoint_path), model=model)
        pl_module.eval()
    else:
        raise ValueError("No checkpoint found")

    # Initialize the LaPlace approximation
    laplace_filename = config.laplace.get_default_laplace_name(model_mode)
    laplace_curv: laplace.ParametricLaplace = None
    if args.checkpoint is True:
        laplace_curv = checkpoint.load_object(laplace_filename, path_args={"save_path": log_path / 'laplace'}, library='dill')
    if laplace_curv is None:
        laplace_params = laplace_config.get_default_laplace_params(model_mode)
        prior_optimization_params = laplace_config.get_default_laplace_prior_optimization_params(model_mode)
        laplace_curv = bayesian_laplace.compute_laplace_for_model(model, train_dataloader, val_dataloader, laplace_params, prior_optimization_params)
        checkpoint.save_object(laplace_curv, laplace_filename, save_path=log_path / 'laplace', library='dill')

    # Evaluate the LaPlace approximation
    laplace_pl_module = laplace_config.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
    laplace_trainer = laplace_config.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": log_path}) 
    # faster_train_dataloader = torch_data.DataLoader(train_dataloader.dataset, batch_size=val_dataloader.batch_size, shuffle=False, num_workers=train_dataloader.num_workers)
    # laplace_trainer.validate(laplace_pl_module, faster_train_dataloader, verbose=args.verbose)
    laplace_trainer.validate(laplace_pl_module, val_dataloader, verbose=args.verbose)

    # laplace_curv.model.eval()
    laplace_pl_module.eval()
    # utils.evaluate_model(nn.Sequential(laplace_curv.model, nn.Softmax(dim=-1)), faster_train_dataloader, "MAP")
    if next(laplace_curv.model.parameters()).device.type != laplace_curv._device:
        laplace_curv.model.to(laplace_curv._device)
    if laplace_pl_module._pred_mode == "deterministic":
        utils.evaluate_model(nn.Sequential(laplace_curv.model, nn.Softmax(dim=-1)), val_dataloader, "MAP")
    elif laplace_pl_module._pred_mode == "bayesian":
        utils.evaluate_model(laplace_curv, val_dataloader, "LaPlace", 
                         pred_type=laplace_pl_module._pred_type, n_samples=laplace_pl_module._n_samples)
    else:
        raise ValueError(f"Unknown prediction mode {laplace_pl_module._pred_mode}")
if __name__ == '__main__':
    main()
        

