from typing import Dict, List, Optional, Text, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import laplace
import laplace.utils.matrix


from util import verification, plot as plot_util
from util.log import histogram as histogram_util
from util.tensor.eigenvalue import NetworkGranularity, ReduceMode, reduce_eigenvalues_per_channel, reduce_eigenvalues_per_layer_channel, collect_eigenvalues_per_layer
from util.tensor.kron import KronBlockDecomposed, from_kron_block_decomposed_to_tensor
from network import lightning as lightning
from util.laplace import compute_laplace_eigenvalues
from util.network import compute_model_decomposition


def log_laplace_eigenvalues(laplace_curv: laplace.ParametricLaplace, logger: TensorBoardLogger, params: List[Tuple[NetworkGranularity, Optional[ReduceMode]]]):
    weight_eigenvalues = compute_laplace_eigenvalues(laplace_curv)
    model_decompostion = compute_model_decomposition(laplace_curv.model)
        
    if isinstance(laplace_curv, laplace.FullLaplace):
        prefix: Text = "laplace_eig/full"
    elif isinstance(laplace_curv, laplace.KronLaplace):
        prefix: Text = "laplace_eig/kron"
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")

    for (x_axis, reduce_mode) in params:
        for (layer_name, fig) in histogram_eigenvalues(weight_eigenvalues, x_axis=x_axis, reduce_mode=reduce_mode, model_decomposition=model_decompostion).items():
            fig_prefix = f"{prefix}/{x_axis}/{reduce_mode}"
            fig_name = f"{fig_prefix}/{layer_name}" if layer_name != '' else fig_prefix
            logger.experiment.add_figure(fig_name, fig)



# TODO: delete function if not used
def heatmap_eigenvalues(weight_eigenvalues: Union[torch.Tensor, KronBlockDecomposed], x_axis: NetworkGranularity= 'weight', reduce_mode: Optional[ReduceMode] = None, model_decomposition: Optional[Dict[Text, List[int]]]=None) -> Dict[Text, figure.Figure]:
    eigval_vmin, eigval_vmax = None, None # 0, 20000
    if isinstance(weight_eigenvalues, list):
        weight_eigenvalues = from_kron_block_decomposed_to_tensor(weight_eigenvalues)

    if isinstance(weight_eigenvalues, torch.Tensor):
        if x_axis == 'weight':
            fig, ax = plt.subplots(figsize=(10,10))
            data = weight_eigenvalues.detach().cpu().numpy()
            data = np.diag(data)
            data = np.where(data == 0, np.nan, data)
            labels = []
            plot_util.heatmap(
                data=data,
                row_labels=labels,
                col_labels=labels, 
                ax=ax,
                vmin=eigval_vmin, vmax=eigval_vmax)   
            return {'': fig}
        elif x_axis == 'channel':
            verification.check_not_none(reduce_mode)
            verification.check_not_none(model_decomposition)
            channel_eigenvalues = reduce_eigenvalues_per_channel(weight_eigenvalues, model_decomposition, reduce_mode)

            fig, ax = plt.subplots(figsize=(10,10))
            data = channel_eigenvalues.detach().cpu().numpy()
            data = np.diag(data)
            data = np.where(data == 0, np.nan, data)
            labels = [f'{param_name}.{channel_idx}' for (param_name, channel_param_counts) in model_decomposition.items() for channel_idx in range(len(channel_param_counts))]
            plot_util.heatmap(
                data=data,
                row_labels=labels,
                col_labels=labels, 
                ax=ax,
                vmin=eigval_vmin, vmax=eigval_vmax)  

            histogram_fig = histogram_util.histogram_vector(channel_eigenvalues.detach().cpu().numpy())

            return {'heatmap': fig, 'histogram': histogram_fig}
        elif x_axis == 'layer_channel':
            verification.check_not_none(reduce_mode)
            verification.check_not_none(model_decomposition)
            layer_channel_eigenvalues = reduce_eigenvalues_per_layer_channel(weight_eigenvalues, model_decomposition, reduce_mode)
            result = {}
            for (layer_name, channel_eigenvalues) in layer_channel_eigenvalues.items():
                fig, ax = plt.subplots(figsize=(10,10))
                data = channel_eigenvalues.detach().cpu().numpy()
                data = np.diag(data)
                data = np.where(data == 0, np.nan, data)
                labels = [f'{layer_name}.{channel_idx}' for channel_idx in range(len(channel_eigenvalues))]
                plot_util.heatmap(
                    data=data,
                    row_labels=labels,
                    col_labels=labels, 
                    ax=ax,
                    vmin=eigval_vmin, vmax=eigval_vmax)   
                result[f'heatmap/{layer_name}']=fig
                histogram_fig = histogram_util.histogram_vector(channel_eigenvalues.detach().cpu().numpy())
                result[f'histogram/{layer_name}'] = histogram_fig
            return result
        elif x_axis == 'layer':
            verification.check_not_none(reduce_mode)
            verification.check_not_none(model_decomposition)
            layer_channel_eigenvalues = reduce_eigenvalues_per_layer_channel(weight_eigenvalues, model_decomposition, reduce_mode)
            result = {}
            fig, ax = plt.subplots(figsize=(10,10))
            data = torch.cat(list(layer_channel_eigenvalues.values())).detach().cpu().numpy()
            data = np.diag(data)
            data = np.where(data == 0, np.nan, data)
            labels = [f'{layer_name}' if channel_idx == 0 else '' 
                      for (layer_name, channel_eigenvalues) in layer_channel_eigenvalues.items()
                      for channel_idx in range(len(channel_eigenvalues))]
            plot_util.heatmap(
                data=data,
                row_labels=labels,
                col_labels=labels, 
                ax=ax,
                vmin=eigval_vmin, vmax=eigval_vmax)   
            histogram_fig = histogram_util.histogram_vector(torch.cat(list(layer_channel_eigenvalues.values())).detach().cpu().numpy())
            return {'heatmap': fig, 'histogram': histogram_fig}
        else:
            raise NotImplementedError(f"Unknown x_axis {x_axis}")
    else:
        raise ValueError(f"Unknown type of eigenvalues {type(weight_eigenvalues)}")


def histogram_eigenvalues(weight_eigenvalues: Union[torch.Tensor, KronBlockDecomposed], x_axis: NetworkGranularity= 'weight', reduce_mode: Optional[ReduceMode] = None, model_decomposition: Optional[Dict[Text, List[int]]]=None) -> Dict[Text, figure.Figure]:
    eigval_vmin, eigval_vmax = None, None # 0, 20000
    if isinstance(weight_eigenvalues, list):
        weight_eigenvalues = from_kron_block_decomposed_to_tensor(weight_eigenvalues)

    if isinstance(weight_eigenvalues, torch.Tensor):
        if x_axis == 'weight':
            data = weight_eigenvalues.detach().cpu().numpy() 
            fig, ax = plt.subplots()
            ax.hist(data)
            return {'': fig}
        # elif x_axis == 'channel':
        #     verification.check_not_none(reduce_mode)
        #     verification.check_not_none(model_decomposition)
        #     channel_eigenvalues = reduce_eigenvalues_per_channel(weight_eigenvalues, model_decomposition, reduce_mode)
        #     data = channel_eigenvalues.detach().cpu().numpy()
        #     fig = histogram_util.histogram_vector(data)
        #     return {'': fig}
        elif x_axis == 'layer_channel':
            verification.check_not_none(model_decomposition)
            all_layer_eigenvalues = collect_eigenvalues_per_layer(weight_eigenvalues, model_decomposition)
            result = {}
            for (layer_name, layer_eigenvalues) in all_layer_eigenvalues.items():
                fig, ax = plt.subplots()
                ax.hist(layer_eigenvalues)
                result[f'histogram/{layer_name}'] = fig
            return result
        elif x_axis == 'layer':
            verification.check_not_none(model_decomposition)
            layer_eigenvalues = collect_eigenvalues_per_layer(weight_eigenvalues, model_decomposition)
            result = {}
            # for (layer_name, channel_eigenvalues) in layer_channel_eigenvalues.items():
            fig, ax = plt.subplots()
            ax.hist(list(layer_eigenvalues.values()), stacked=True, label=list(layer_eigenvalues.keys()))
            ax.legend()
            return {'': fig}
        else:
            raise NotImplementedError(f"Unknown x_axis {x_axis}")
    else:
        raise ValueError(f"Unknown type of eigenvalues {type(weight_eigenvalues)}")
