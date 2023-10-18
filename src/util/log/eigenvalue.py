from typing import Dict, Iterable, List, Optional, Text, Union, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import laplace
import laplace.utils.matrix


from util import verification, assertion, plot as plot_util
from util.log import histogram as histogram_util
from util.tensor.eigenvalue import NetworkGranularity, ReduceMode, reduce_eigenvalues_per_channel, reduce_eigenvalues_per_layer_channel, collect_eigenvalues_per_layer
from util.tensor.kron import KronBlockDecomposed, from_kron_block_decomposed_to_tensor, from_kron_block_decomposed_to_tensors, kron_block_decomposed_max, kron_block_decomposed_min
from network import lightning as lightning
from util.laplace import compute_laplace_eigenvalues, compute_laplace_eigendecomp
from util.network import compute_model_decomposition


def pca_mask(eigenvalues: KronDecomposed, top: Optional[Union[float, int]]=None, coverage: Optional[float]=None) -> List[int]:
    if top is None and coverage is None:
        raise ValueError("Either top or coverage must be specified")
    elif top is not None and coverage is not None:
        raise ValueError("Only one of top or coverage must be specified")
    
    if top is not None:
        if isinstance(top, float):
            top = int(top * len(eigenvalues))
        # Get the indices of the top eigenvalues
        top_eigenvalues = torch.topk(eigenvalues, top, largest=True)
        top_eigenvalue_indices = top_eigenvalues.indices

    elif coverage is not None:
        # Get the indices of the eigenvalues whose value summed cover (coverage)% of the total sum
        sorted_eigenvalues, sorted_indices = torch.sort(eigenvalues, descending=True)
        sorted_eigenvalues_sum = torch.sum(sorted_eigenvalues)
        sorted_eigenvalues_cumsum = torch.cumsum(sorted_eigenvalues, dim=0)
        sorted_eigenvalues_cumsum_coverage = sorted_eigenvalues_cumsum / sorted_eigenvalues_sum
        top_eigenvalue_indices = sorted_indices[sorted_eigenvalues_cumsum_coverage <= coverage]

    else:
        raise ValueError("Either top or coverage must be specified")

    return top_eigenvalue_indices

def log_laplace_eigenvalues(laplace_curv: laplace.ParametricLaplace, logger: TensorBoardLogger, params: List[Tuple[NetworkGranularity, Optional[ReduceMode]]]):
    weight_eigenvalues, weight_eigenvectors = compute_laplace_eigendecomp(laplace_curv)
    model_decompostion = compute_model_decomposition(laplace_curv.model)

    print(f"Finding the most important principal components(eigenvectors) of the Laplace approximation")
    print(f)
    top_eigenvector_indices = pca_mask(weight_eigenvalues, coverage=0.95)
    top_eigenvectors = weight_eigenvectors[:, top_eigenvector_indices]
    top_eigenvalues = weight_eigenvalues[top_eigenvector_indices]
    print(f"From {len(weight_eigenvalues)} eigenvalues and eigenvectors, {len(top_eigenvalues)} are selected")

    return
        
    if isinstance(laplace_curv, laplace.FullLaplace):
        prefix: Text = "laplace_eig/full"
    elif isinstance(laplace_curv, laplace.KronLaplace):
        prefix: Text = "laplace_eig/kron"
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")

    for (x_axis, reduce_mode) in params:
        for (layer_name, fig) in histogram_eigenvalues(weight_eigenvalues, x_axis=x_axis, reduce_mode=reduce_mode, model_decomposition=model_decompostion).items():
            fig_prefix = f"{prefix}/hist/{x_axis}/{reduce_mode}"
            fig_name = f"{fig_prefix}/{layer_name}" if layer_name != '' else fig_prefix
            logger.experiment.add_figure(fig_name, fig)

    pca_loadings = compute_pca_loadings(weight_eigenvalues, weight_eigenvectors)
    fig_prefix = f"{prefix}/pca_loadings"
    for (fig_title, heatmap_figure) in heatmap_loadings(pca_loadings, model_decomposition=model_decompostion):
        print(f"Logging figure {fig_prefix}/{fig_title}")
        fig_name = f"{fig_prefix}/{fig_title}" if fig_title != '' else fig_prefix
        logger.experiment.add_figure(fig_name, heatmap_figure)


def compute_pca_loadings(eigenvalues, eigenvectors) -> Union[torch.Tensor, KronBlockDecomposed]:
    # eigenvalues, eigenvectors = compute_laplace_eigendecomp(laplace_curv)
    if isinstance(eigenvalues, torch.Tensor) and isinstance(eigenvectors, torch.Tensor):
        loadings = torch.matmul(eigenvectors, torch.diag(torch.sqrt(eigenvalues)))
    # If eigenvalues and eigenvectors are KronBlockDecomposed
    elif isinstance(eigenvalues, list) and isinstance(eigenvectors, list):
        loadings: KronBlockDecomposed = []
        for (eigenvalue_decomposed, eigenvector_decomposed) in zip(eigenvalues, eigenvectors):
            loading = tuple([torch.matmul(eigenvector, torch.diag(torch.sqrt(eigenvalue))) for (eigenvalue, eigenvector) in zip(eigenvalue_decomposed, eigenvector_decomposed)])
            loadings.append(loading)

            # assertion.assert_tensor_close(from_kron_block_decomposed_to_tensor([loading]),
            #                               torch.matmul(from_kron_block_decomposed_to_tensor([eigenvector_decomposed]), torch.diag(torch.sqrt(from_kron_block_decomposed_to_tensor([eigenvalue_decomposed])))))

    else:
        raise ValueError(f"Unknown type of eigenvalues {type(eigenvalues)} and eigenvectors {type(eigenvectors)}")
    
    return loadings



def heatmap_loadings(loadings: Union[torch.Tensor, KronBlockDecomposed], x_axis: NetworkGranularity= 'weight', reduce_mode: Optional[ReduceMode] = None, model_decomposition: Optional[Dict[Text, List[int]]]=None, threshold: float = 0.5) -> Iterable[Tuple[Text, figure.Figure]]:
    if isinstance(loadings, list):
        loadings_shape = [[list(loading.shape) for loading in layer_loadings] for layer_loadings in loadings]
    else:
        loadings_shape = [list(loadings.shape)]
    print(f"Preparing heatmap for loading of shape {loadings_shape}")
    if isinstance(loadings, list):
        vmin = kron_block_decomposed_min(loadings)
        vmax = kron_block_decomposed_max(loadings)
        for (layer_name, layer_loadings) in zip(model_decomposition.keys(), from_kron_block_decomposed_to_tensors(loadings)):
            print(f"\tPreparing heatmap for layer {layer_name}")
            fig, ax = plt.subplots(figsize=(10,10))
            layer_loadings = layer_loadings.detach()
            # print(f"\tMoving data of shape {data.shape} to cpu as numpy array")
            # data = layer_loadings.detach().cpu().numpy()
            if len(layer_loadings.shape) == 1:
                layer_loadings = torch.diag(layer_loadings)
            # print(f"\tReplacing absolute values below threshold {threshold} with NaNs")
            layer_loadings[layer_loadings.abs() <= threshold] = torch.nan


            if x_axis == 'layer_channel':
                verification.check_not_none(reduce_mode)
                verification.check_not_none(model_decomposition)
                layer_channel_eigenvalues = reduce_loadings_per_layer_channel(layer_loadings, model_decomposition, reduce_mode)
                result = {}
                for (layer_name, channel_eigenvalues) in layer_channel_eigenvalues.items():
                    fig, ax = plt.subplots(figsize=(10,10))
                    data = channel_eigenvalues.detach().cpu().numpy()
                    if len(data.shape) == 1:
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

            # print(f"\tAsserting that all values are between {vmin} and {vmax}")
            # assertion.assert_le(vmin, np.nanmin(data))
            # assertion.assert_le(np.nanmax(data), vmax)
            # Cumsum the number of parameters per channel
            # print(f"\tCumsumming the number of parameters per channel to identify channel indices")
            channel_cumsum = np.cumsum([0] + model_decomposition[layer_name])
            labels = {channel_cumsum[i]: f'{layer_name}.{i}' for i in range(len(channel_cumsum))}
            # labels = [f'{layer_name}.{channel_idx}' for channel_idx in range(len(layer_loadings))]
            # print(f"\tPassing data to heatmap of shape {data.shape} with labels {labels} to be plotted")
            plot_util.heatmap(
                data=data,
                row_labels=labels,
                col_labels=[], 
                vmin=vmin, vmax=vmax,
                ax=ax)   
            # print(f"\tYielding heatmap for layer {layer_name}")
            yield layer_name, fig
    elif isinstance(loadings, torch.Tensor):
        fig, ax = plt.subplots(figsize=(10,10))
        data = loadings.detach().cpu().numpy()
        if len(data.shape) == 1:
            data = np.diag(data)
        data = np.where(data == 0, np.nan, data)
        labels = []
        plot_util.heatmap(
            data=data,
            row_labels=labels,
            col_labels=labels, 
            ax=ax)   
        yield '', fig
    else:
        raise ValueError(f"Unknown type of loadings {type(loadings)}")
    
# TODO: delete function if not used
def heatmap_eigenvalues(weight_eigenvalues: Union[torch.Tensor, KronBlockDecomposed], x_axis: NetworkGranularity= 'weight', reduce_mode: Optional[ReduceMode] = None, model_decomposition: Optional[Dict[Text, List[int]]]=None) -> Dict[Text, figure.Figure]:
    if isinstance(weight_eigenvalues, list):
        weight_eigenvalues = from_kron_block_decomposed_to_tensor(weight_eigenvalues)
    eigval_vmin, eigval_vmax = weight_eigenvalues.min(), weight_eigenvalues.max()

    if isinstance(weight_eigenvalues, torch.Tensor):

        if x_axis == 'weight':
            fig, ax = plt.subplots(figsize=(10,10))
            data = weight_eigenvalues.detach().cpu().numpy()
            if len(data.shape) == 1:
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
            if len(data.shape) == 1:
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
                if len(data.shape) == 1:
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
            if len(data.shape) == 1:
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
