
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

from util import assertion, data as data_utils, checkpoint
from util import lightning as lightning_util
import config
from network.bayesian.laplace import laplace as bayesian_laplace
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

    parser.add_argument('--data', help='specify which dataset to use', type=str, choices=config.mode.AVAILABLE_DATASETS, default='cifar10', required=False)
    parser.add_argument('--model', help='specify which model to use', type=str, choices=config.mode.AVAILABLE_MODELS, default='resnet20', required=False)

    parser.add_argument('--checkpoint', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.set_defaults(checkpoint=False)

    parser.add_argument('--eval', help='specify if a pre saved version of the laplace should be used', action='store_true')
    parser.add_argument('--no-eval', dest='eval', action='store_false')
    parser.set_defaults(eval=True)

    parser.add_argument('--device', help='specify which devices to use', type=int, default=0, required=False)

    return parser.parse_args()

def main():
    args: argparse.Namespace = parse_args()

    # Set the device
    torch.cuda.set_device(args.device)

    model_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "model"
    log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "bayesian"
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
    data_module = config.bayesian.laplace.data.lightning.get_default_datamodule(data_mode)
    utils.verbose_and_log(f"Datamodule initialized: \n{data_utils.verbose_datamodule(data_module)}", args.verbose, args.log)

    # Initialize the model
    model = config.network.get_default_model(model_mode)
    utils.verbose_and_log(f"Model created: {model}", args.verbose, args.log)

    pl_module: pl.LightningModule = config.network.lightning.get_default_lightning_module(model_mode, model)
    best_checkpoint_path = checkpoint.find_best_checkpoint(model_log_path)
    if best_checkpoint_path is not None:
        pl_module.__class__.load_from_checkpoint(str(best_checkpoint_path), model=model)
        pl_module.eval()
    else:
        pretrained_path = checkpoint.find_pretrained(model_log_path)
        if pretrained_path is not None:
            utils.verbose_and_log(f"Loading pretrained model from {pretrained_path}", args.verbose, args.log)
            checkpoint.load_model(model, file_name=pretrained_path.stem, 
                                path_args={'save_path': pretrained_path.parent, 'file_ext': pretrained_path.suffix[1:]})
        else:   
            raise ValueError("No checkpoint or pretrained model found")



    # Initialize the LaPlace approximation
    laplace_filename = config.bayesian.laplace.get_default_laplace_name(model_mode)
    laplace_curv: Optional[laplace.ParametricLaplace] = None
    if args.checkpoint is True:
        utils.verbose_and_log(f"Loading LaPlace approximation from {laplace_filename}", args.verbose, args.log)
        laplace_curv_path: Optional[Path] = None
        laplace_curv, laplace_curv_path = checkpoint.load_object(laplace_filename, path_args={"save_path": log_path / 'laplace'}, library='dill', with_path=True)
        if laplace_curv_path is not None:
            utils.verbose_and_log(f"Loaded LaPlace approximation from {laplace_curv_path}", args.verbose, args.log)
    if laplace_curv is None:
        utils.verbose_and_log(f"Computing LaPlace approximation", args.verbose, args.log)
        laplace_params = config.bayesian.laplace.get_default_laplace_params(model_mode)
        prior_optimization_params = config.bayesian.laplace.get_default_laplace_prior_optimization_params(model_mode)
        data_module.setup(stage="fit")
        laplace_curv = bayesian_laplace.compute_laplace_for_model(model, data_module.train_dataloader(), data_module.val_dataloader(), laplace_params, prior_optimization_params, verbose=args.verbose)
        utils.verbose_and_log(f"Saving LaPlace approximation to {laplace_filename}", args.verbose, args.log)
        checkpoint.save_object(laplace_curv, laplace_filename, save_path=log_path / 'laplace', library='dill')

    if args.eval:
        # Evaluate the LaPlace approximation
        laplace_pl_module = config.bayesian.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
        laplace_trainer = config.bayesian.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": log_path}) 
        
        utils.make_deterministic(False)

        utils.verbose_and_log(f"Original deterministic metrics", args.verbose, args.log)
        laplace_trainer.validate(pl_module, data_module, verbose=args.verbose)

        utils.verbose_and_log(f"LaPlace metrics", args.verbose, args.log)
        laplace_trainer.validate(laplace_pl_module, data_module, verbose=args.verbose)

        laplace_curv.model.eval()
        laplace_pl_module.eval()
        if next(laplace_curv.model.parameters()).device.type != laplace_curv._device:
            laplace_curv.model.to(laplace_curv._device)
        if laplace_pl_module._pred_mode == "deterministic":
            utils.evaluate_model(nn.Sequential(laplace_curv.model, nn.Softmax(dim=-1)), data_module.val_dataloader(), "MAP")
        elif laplace_pl_module._pred_mode == "bayesian":
            utils.evaluate_model(laplace_curv, data_module.val_dataloader(), "LaPlace", 
                            pred_type=laplace_pl_module._pred_type, n_samples=laplace_pl_module._n_samples)
        else:
            raise ValueError(f"Unknown prediction mode {laplace_pl_module._pred_mode}")

# Register the cleanup function to be called on exit
utils.register_cleanup()



if __name__ == '__main__':
    utils.limit_memory(9 * 1024 * 1024 * 1024)
    utils.catch_and_print(lambda: utils.run_with_timeout(main, timeout=20*60))
