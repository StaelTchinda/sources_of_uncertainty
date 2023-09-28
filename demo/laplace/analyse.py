
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Text, Union
import logging
import lightning.pytorch as pl
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
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

from util import assertion, checkpoint, verification, data as data_utils
from util import lightning as lightning_util, plot as plot_util
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils
import metrics
import config



class SaveLayerVarianceCallback(pl.Callback):
    def __init__(self, stage: Literal['train', 'val', 'test'] = 'val'):
        self.stage = stage
        self._sampling_index: Dict[nn.Module, int] = {}
        self._batch_index: Dict[nn.Module, int] = {}
        self._variances: Dict[nn.Module, metrics.VarianceEstimator] = {}

    def on_validation_start(self, trainer, pl_module):
        print("Validation is starting")
        if self.stage == 'val':
            assert isinstance(pl_module, lightning.laplace.LaplaceModule)
            self.register_hooks(pl_module.laplace.model)
            pl_module.laplace.model.register_forward_hook(lambda module, input, output: self.module_hook_fn(module, input, torch.softmax(output, dim=-1)))

    def on_validation_end(self, trainer, pl_module):
        print("Validation is ending")

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.stage == 'val':
            self.globally_set_batch_index(batch_idx)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def register_hooks(self, module: nn.Module):
        for sub_module in module.modules():
            if not hasattr(sub_module, 'weight'):
                continue
            sub_module.register_forward_hook(self.module_hook_fn)

    def module_hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if module not in self._variances:
            self._variances[module] = metrics.VarianceEstimator()
        if module not in self._sampling_index:
            self._sampling_index[module] = 0
        if module not in self._batch_index:
            self._batch_index[module] = 0
        self._variances[module].feed_probs(output, gt_labels=None, samples=None,
                                           sampling_index=self._sampling_index[module], 
                                           batch_index=self._batch_index[module])
        self.increase_sampling_index(module)

    def get_layer_variance(self, module: nn.Module):
        if module not in self._variances:
            return None
        return self._variances[module].get_metric_value()
    
    def set_batch_index(self, module: nn.Module, batch_index: int):
        self._batch_index[module] = batch_index
    
    def globally_set_batch_index(self, batch_index: int):
        for module in self._batch_index.keys():
            self.set_batch_index(module, batch_index)

    def increase_batch_index(self, module: nn.Module):
        self._batch_index[module] += 1

    def globally_increase_batch_index(self):
        for module in self._batch_index.keys():
            self.increase_batch_index(module)

    def set_sampling_index(self, module: nn.Module, sampling_index: int):
        self._sampling_index[module] = sampling_index

    def globally_set_sampling_index(self, sampling_index: int):
        for module in self._sampling_index.keys():
            self.set_sampling_index(module, sampling_index)
    
    def increase_sampling_index(self, module: nn.Module):
        self._sampling_index[module] += 1

    def globally_increase_sampling_index(self):
        for module in self._sampling_index.keys():
            self.increase_sampling_index(module)
    
class LayerVarianceHook:
    def __init__(self):
        self._sampling_index: Dict[nn.Module, int] = {}
        self._batch_index: Dict[nn.Module, int] = {}
        self._variances: Dict[nn.Module, metrics.VarianceEstimator] = {}

    def module_hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if module not in self._variances:
            self._variances[module] = metrics.VarianceEstimator()
        if module not in self._sampling_index:
            self._sampling_index[module] = 0
        if module not in self._batch_index:
            self._batch_index[module] = 0
        self._variances[module].feed_probs(output, gt_labels=None, samples=None,
                                           sampling_index=self._sampling_index[module], 
                                           batch_index=self._batch_index[module])
        self._sampling_index[module] += 1

    def get_module_std_dev(self, module: nn.Module):
        if module not in self._variances:
            return None
        return self._variances[module].get_metric_value()
    
    def increase_batch_index(self, module: nn.Module):
        self._batch_index[module] += 1

    def generally_increase_batch_index(self):
        for module in self._batch_index.keys():
            self.increase_batch_index(module)


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

    # Goal 1: anaylse the evolution of the variance over the layers
    # Approach:
    # - add a/multiple hook(s) to the model, which 
    #    - computes the variance of the output of each layer
    #    - saves it
    #    - logs it
    # - run the model on the validation set
    variance_callback = SaveLayerVarianceCallback()

    # Initialize the LaPlace module
    laplace_pl_module = config.laplace.lightning.get_default_lightning_laplace_module(model_mode, laplace_curv)
    laplace_trainer = config.laplace.lightning.get_default_lightning_laplace_trainer(
        model_mode, 
        {
            "default_root_dir": log_path,
            "callbacks": [variance_callback]
        }
    )

    laplace_trainer.validate(laplace_pl_module, data_module)
    if args.verbose:
        print("Finished validation")
        print(f"Variances per layer")
        for module, variance in variance_callback._variances.items():
            print(f"{module} : {variance.get_metric_value()}")


if __name__ == '__main__':
    main()
        

