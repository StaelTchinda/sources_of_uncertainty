
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

from util.log.eigenvalue import log_laplace_eigenvalues

def add_src_to_path(demo_dir_path: Optional[Path] = None):
    import sys, pathlib
    if demo_dir_path is None:
        demo_dir_path = pathlib.Path(__file__).parent.resolve()
    src_dir_path: Path = demo_dir_path.parent.parent / "src"

    sys.path.append(str(src_dir_path))


add_src_to_path()

from util import assertion, config, checkpoint, verification
from util import lightning as lightning_util, plot as plot_util
from config import laplace as laplace_config, network as network_config, metrics as metrics_config
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils
import metrics



class LayerVarianceHook:
    def __init__(self, top_k: Union[int, Literal['all']] = 'all'):
        self._top_k: Union[int, Literal['all']] = top_k
        self._std_dev_per_samples: Dict[nn.Module, List[torch.Tensor]] = {}

    def module_hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        output_std_dev = metrics.standard_dev(output, top_k=self._top_k)
        if module not in self._std_dev_per_samples:
            self._std_dev_per_samples[module] = [output_std_dev]
        else:
            self._std_dev_per_samples[module].append(output_std_dev)

    def get_module_std_dev(self, module: nn.Module):
        if module not in self._std_dev_per_samples:
            return None
        std_dev_per_samples = torch.cat(self._std_dev_per_samples[module])
        std_dev = std_dev_per_samples.mean()
        return std_dev

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


    # Goal 1: anaylse the evolution of the variance over the layers
    # Approach:
    # - add a/multiple hook(s) to the model, which 
    #    - computes the variance of the output of each layer
    #    - saves it
    #    - logs it
    # - run the model on the validation set


    # Initialize the LaPlace module
    laplace_pl_module = laplace_config.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
    laplace_trainer = laplace_config.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": log_path})


    # Goal 2: observe the eigenvalues of the Hessian
    log_laplace_eigenvalues(laplace_curv, laplace_trainer.logger, params=[
        ('weight', None), 
        ('channel', 'max'), ('channel', 'min'), ('channel', 'mean'), ('layer_channel', 'max'), 
        ('layer_channel', 'min'), ('layer_channel', 'mean'), 
        ('layer', 'max'), ('layer', 'min'), ('layer', 'mean')
        ])

    # Print the variance per layer
    # for module, std_dev in layer_variance_hook._std_dev_per_samples.items():
        # utils.verbose_and_log(f"Std dev of {module} is {std_dev}", args.verbose, args.log)
if __name__ == '__main__':
    main()
        

