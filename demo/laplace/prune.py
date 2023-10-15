
import argparse
import copy
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
from network.bayesian import laplace as bayesian_laplace
from network import lightning as lightning
from util import utils

from network.pruning import pruner as pruning_wrapper, util as pruning_util



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

    parser.add_argument('--strategy', help='specify which pruning strategy should be used', nargs='+', choices=pruning_wrapper.AVAILABLE_PRUNING_STRATEGIES, default=pruning_wrapper.AVAILABLE_PRUNING_STRATEGIES, required=False)
    parser.add_argument('--sparsity', help='specify which sparsity the module should have to have', nargs='*', default=None, required=False)
    parser.add_argument('--layer', help='specify which layers should be pruned we want the model to have', nargs='*', default=None, required=False)

    return parser.parse_args()


def main():
    args: argparse.Namespace = parse_args()

    laplace_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "laplace"
    prune_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "laplace" / "analyse" / "layer" / "prune"
    log_foldername: Text = f"run {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

    if args.log:
        basic_config_params = config.log.get_log_basic_config(filename=laplace_log_path / 'prune' / f'{log_foldername}.log')
        logging.basicConfig(**basic_config_params)
        utils.verbose_and_log(f"Logging enabled: {log_foldername}", args.verbose, args.log)
    else:
        logging.disable(logging.CRITICAL)

    data_mode: config.mode.DataMode = args.data
    model_mode: config.mode.ModelMode = args.model

    # Initialize the dataloaders
    data_module = config.data.lightning.get_default_datamodule(data_mode)
    utils.verbose_and_log(f"Datamodule initialized: \n{data_utils.verbose_datamodule(data_module)}", args.verbose, args.log)

    # Initialize the laplace approximation
    laplace_filename = config.laplace.get_default_laplace_name(model_mode)
    utils.verbose_and_log(f"Loading LaPlace approximation with name {laplace_filename} from {laplace_log_path}", args.verbose, args.log)
    laplace_curv: laplace.ParametricLaplace = checkpoint.load_object(laplace_filename, path_args={"save_path": laplace_log_path}, library='dill')
    original_laplace_curv = copy.deepcopy(laplace_curv)

    if laplace_curv is None:
        raise ValueError("No laplace approximation found")
    utils.verbose_and_log(f"Laplace loaded to model: {laplace_curv.model}", args.verbose, args.log)

    laplace_pl_module = config.laplace.lightning.get_default_lightning_laplace_pruning_module(model_mode, laplace_curv) 
    
    pruning_layer_names: List[Text] = []
    if args.layer is not None:
        pruning_layer_names = args.layer
    else:
        pruning_layer_names = list(pruning_util.get_prunable_named_modules(laplace_pl_module.laplace.model).keys())

    pruning_sparsities: List[float] = []
    if args.sparsity is not None:
        pruning_sparsities = args.sparsity
    else:
        pruning_sparsities = config.prune.get_default_sparsities(data_mode, model_mode)
    pruning_amounts = pruning_util.get_required_amounts_for_sparsities(pruning_sparsities)
    utils.verbose_and_log(f"To prune the model to the sparsities {pruning_sparsities}, it's necessary to prune it at the amounts {pruning_amounts}", args.verbose, args.log)

    for strategy in args.strategy:
        laplace_pl_module.pruner.pruning_strategy = strategy

        for layer in pruning_layer_names:
            laplace_pl_module.pruner.module_name_to_prune = layer

            prune_log_path: Path = config.path.CHECKPOINT_PATH / f"{args.data}" / f"{args.model}" / "prune" / log_foldername / f"{strategy}" / f"{layer}"
            laplace_trainer = config.laplace.lightning.get_default_lightning_laplace_trainer(model_mode, {"default_root_dir": prune_log_path})

            for (sparsity, amount) in zip(pruning_sparsities, pruning_amounts):
                utils.verbose_and_log(f"Pruning {layer} with strategy {strategy} and amount {amount}", args.verbose, args.log)
                laplace_pl_module.pruner.pruning_amount = amount
                laplace_pl_module.pruner.pruning_sparsity = sparsity
                laplace_pl_module.pruner.prune(permanent=True)

                assertion.assert_is(laplace_pl_module.pruner.original_model, laplace_pl_module.laplace.backend.model)

                laplace_trainer.validate(laplace_pl_module, data_module)

                # Log details of sparsity for verification. It's important to log after the validation for the logger object to be initialized.
                model_sparsity = pruning_util.measure_modular_sparsity(laplace_pl_module.laplace.model)
                utils.verbose_and_log(pruning_util.verbose_modular_sparsity(laplace_pl_module.laplace.model, model_sparsity), args.verbose, args.log)
                laplace_pl_module.logger.experiment.add_scalar(f'sparsity/amount', int(amount*100), int(sparsity*100))
                for (module_name, sparsity_counts) in model_sparsity.items():
                    laplace_pl_module.logger.experiment.add_scalar(f'sparsity/sparsity/{module_name}', sparsity_counts[0]/sparsity_counts[1], int(sparsity*100))
                    laplace_pl_module.logger.experiment.add_scalar(f'sparsity/inactive_weight_count/{module_name}', sparsity_counts[0], int(sparsity*100))

            if sparsity != 0.0:
                laplace_pl_module.pruner.unprune()  
                laplace_pl_module.laplace = copy.deepcopy(original_laplace_curv)
                laplace_pl_module.pruner = pruning_wrapper.Pruner(laplace_pl_module.laplace.model, pruning_strategy=strategy)
    
import atexit
# Register the cleanup function to be called on exit
atexit.register(utils.cleanup)

if __name__ == '__main__':
    try:
        utils.make_deterministic()
        main()
    except Exception as e:
        # Handle exceptions gracefully, log errors, etc.
        print("An error occurred:", str(e))
        # Print stacktrace
        import traceback
        traceback.print_exc()        

