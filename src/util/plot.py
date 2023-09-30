from typing import Optional, Union, Text

import torch
from matplotlib import pyplot as plt
from matplotlib import figure
import numpy as np
from typing import Dict, List, Optional, Text, Union
import torch

import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np

from util import verification, log as log_util
from util import plot as plot_util
from network import lightning as lightning
from util.tensor.eigenvalue import NetworkGranularity, ReduceMode, reduce_eigenvalues_per_channel, reduce_eigenvalues_per_layer_channel
from util.tensor.kron import KronBlockDecomposed, from_kron_block_decomposed_to_tensor
import util

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

            histogram_fig = log_util.histogram.histogram_vector(channel_eigenvalues.detach().cpu().numpy())

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
                histogram_fig = log_util.histogram.histogram_vector(channel_eigenvalues.detach().cpu().numpy())
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
            histogram_fig = log_util.histogram.histogram_vector(torch.cat(list(layer_channel_eigenvalues.values())).detach().cpu().numpy())
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
            fig = log_util.histogram.histogram_vector(data)
            return {'': fig}
        elif x_axis == 'channel':
            verification.check_not_none(reduce_mode)
            verification.check_not_none(model_decomposition)
            channel_eigenvalues = reduce_eigenvalues_per_channel(weight_eigenvalues, model_decomposition, reduce_mode)
            data = channel_eigenvalues.detach().cpu().numpy()
            fig = log_util.histogram.histogram_vector(data)
            return {'': fig}
        elif x_axis == 'layer_channel':
            verification.check_not_none(reduce_mode)
            verification.check_not_none(model_decomposition)
            layer_channel_eigenvalues = reduce_eigenvalues_per_layer_channel(weight_eigenvalues, model_decomposition, reduce_mode)
            result = {}
            for (layer_name, channel_eigenvalues) in layer_channel_eigenvalues.items():
                fig, ax = plt.subplots(figsize=(10,10))
                data = channel_eigenvalues.detach().cpu().numpy()
                histogram_fig = log_util.histogram.histogram_vector(data)
                result[f'{layer_name}'] = histogram_fig
            return result
        elif x_axis == 'layer':
            verification.check_not_none(reduce_mode)
            verification.check_not_none(model_decomposition)
            layer_channel_eigenvalues = reduce_eigenvalues_per_layer_channel(weight_eigenvalues, model_decomposition, reduce_mode)
            result = {}
            data = torch.cat(list(layer_channel_eigenvalues.values())).detach().cpu().numpy()
            histogram_fig = log_util.histogram.histogram_vector(data)
            return {'': histogram_fig}
        else:
            raise NotImplementedError(f"Unknown x_axis {x_axis}")
    else:
        raise ValueError(f"Unknown type of eigenvalues {type(weight_eigenvalues)}")

# From https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    default_kwargs = {'cmap': 'jet', 'interpolation': 'nearest'}
    kwargs = {**default_kwargs, **kwargs}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


