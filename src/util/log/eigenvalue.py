from typing import Dict, Iterable, List, Optional, Text, Union, Tuple
from typing import overload

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import figure
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import laplace
import laplace.utils.matrix


from util import verification, assertion, plot as plot_util
# from util.log import histogram as histogram_util
from util.tensor.eigenvalue import NetworkGranularity, ReduceMode, reduce_eigenvalues_per_channel, reduce_eigenvalues_per_layer_channel, collect_eigenvalues_per_layer, reduce_loadings_per_layer_channel
from util.tensor.kron import KronBlock, KronBlockDecomposed, from_kron_block_decomposed_to_tensor, from_kron_block_decomposed_to_tensors, kron_block_decomposed_max, kron_block_decomposed_min
from network import lightning as lightning
from util.laplace import compute_laplace_eigenvalues, compute_laplace_eigendecomp
from util.network import compute_model_decomposition

@overload
def mask_pca_eigenvalues(eigenvalues: torch.Tensor, 
                         mask: List[int]) -> torch.Tensor:
    ...

@overload
def mask_pca_eigenvalues(eigenvalues: KronBlock, 
                         mask: Union[List[int], List[Tuple[int, int]]]) -> KronBlock:
    ...
    
@overload
def mask_pca_eigenvalues(eigenvalues: KronBlockDecomposed,
                         mask: List[Union[List[int], List[Tuple[int, int]]]]) -> KronBlockDecomposed:
    ...

def mask_pca_eigenvalues(eigenvalues, mask):
    if isinstance(eigenvalues, torch.Tensor):
        verification.check_is_instance(mask, list)
        verification.check_is_instance(mask[0], int)
        masked_eigenvalues: torch.Tensor = eigenvalues[mask]
        assertion.assert_le(masked_eigenvalues.sum().item(), eigenvalues.sum().item())
        return masked_eigenvalues
    elif isinstance(eigenvalues, tuple) or \
        (isinstance(eigenvalues, list) and 1<=len(eigenvalues)<=2 and isinstance(eigenvalues[0], torch.Tensor)):
        if len(eigenvalues) == 2:
            verification.check_is_instance(mask, list)
            verification.check_is_instance(mask[0], tuple)
            # Select all unique elements based on indices from the mask
            unique_indices: Tuple[List[int], List[int]] = tuple([np.unique([m[i] for m in mask]) for i in range(2)])
            masked_eigenvalues: KronBlock = tuple([eigenvalues[i][unique_indices[i]] for i in range(2)])
        elif len(eigenvalues) == 1:
            verification.check_is_instance(mask, torch.Tensor)
            masked_eigenvalues: KronBlock = tuple([eigenvalues[0][mask]])
        # verification.check_equals(len(eigenvalues), len(mask))
        assertion.assert_le(torch.cat(list(from_kron_block_decomposed_to_tensors([masked_eigenvalues]))).sum().item(), 
            torch.cat(list(from_kron_block_decomposed_to_tensors([eigenvalues]))).sum().item())
        return masked_eigenvalues
    elif isinstance(eigenvalues, list):
        verification.check_is_instance(mask, list)
        verification.check_equals(len(eigenvalues), len(mask))
        masked_eigenvalues: KronBlockDecomposed = [mask_pca_eigenvalues(eigenvalue, kron_block_mask) for (eigenvalue, kron_block_mask) in zip(eigenvalues, mask)]
        assertion.assert_le(torch.cat(list(from_kron_block_decomposed_to_tensors(masked_eigenvalues))).sum().item(), 
            torch.cat(list(from_kron_block_decomposed_to_tensors(eigenvalues))).sum().item())
        return masked_eigenvalues


@overload
def mask_pca_eigenvectors(eigenvectors: torch.Tensor, 
                         mask: List[int]) -> torch.Tensor:
    ...

@overload
def mask_pca_eigenvectors(eigenvectors: KronBlock, 
                         mask: Union[List[int], List[Tuple[int, int]]]) -> KronBlock:
    ...
    
@overload
def mask_pca_eigenvectors(eigenvectors: KronBlockDecomposed,
                         mask: List[Union[List[int], List[Tuple[int, int]]]]) -> KronBlockDecomposed:
    ...

def mask_pca_eigenvectors(eigenvectors, mask):
    if isinstance(eigenvectors, torch.Tensor):
        verification.check_is_instance(mask, list)
        verification.check_is_instance(mask[0], int)
        masked_eigenvectors: torch.Tensor = eigenvectors[mask]
        return masked_eigenvectors
    elif isinstance(eigenvectors, tuple) or \
        (isinstance(eigenvectors, list) and 1<=len(eigenvectors)<=2 and isinstance(eigenvectors[0], torch.Tensor)):
        if len(eigenvectors) == 2:
            verification.check_is_instance(mask, list)
            verification.check_is_instance(mask[0], tuple)
            # Select all unique elements based on indices from the mask
            unique_indices: Tuple[List[int], List[int]] = tuple([np.unique([m[i] for m in mask]) for i in range(2)])
            masked_eigenvectors: KronBlock = tuple([eigenvectors[i][:,unique_indices[i]] for i in range(2)])
        elif len(eigenvectors) == 1:
            verification.check_is_instance(mask, torch.Tensor)
            masked_eigenvectors: KronBlock = tuple([eigenvectors[0][:,mask]])
        # verification.check_equals(len(eigenvectors), len(mask))
        return masked_eigenvectors
    elif isinstance(eigenvectors, list):
        verification.check_is_instance(mask, list)
        verification.check_equals(len(eigenvectors), len(mask))
        masked_eigenvectors = [mask_pca_eigenvectors(eigenvector, kron_block_mask) for (eigenvector, kron_block_mask) in zip(eigenvectors, mask)]
        return masked_eigenvectors



@overload
def pca_mask(eigenvalues: torch.Tensor, 
             coverage: float) -> List[int]:
    ...

@overload
def pca_mask(eigenvalues: KronBlock, 
             coverage: float) -> Union[List[int], List[Tuple[int, int]]]:
    ...
    
@overload
def pca_mask(eigenvalues: KronBlockDecomposed, 
             coverage: float) -> List[Union[List[int], List[Tuple[int, int]]]]:
    ...

def pca_mask(eigenvalues, coverage):
    if isinstance(eigenvalues, torch.Tensor):
        # Get the indices of the eigenvalues whose value summed cover (coverage)% of the total sum
        sorted_eigenvalues, sorted_indices = torch.sort(eigenvalues, descending=True)
        sorted_eigenvalues_sum = torch.sum(sorted_eigenvalues)
        sorted_eigenvalues_cumsum = torch.cumsum(sorted_eigenvalues, dim=0)
        sorted_eigenvalues_cumsum_coverage = sorted_eigenvalues_cumsum / sorted_eigenvalues_sum
        top_eigenvalue_indices = sorted_indices[sorted_eigenvalues_cumsum_coverage <= coverage]
    # elif isinstance(eigenvalues, KronBlock):
    elif isinstance(eigenvalues, tuple) or \
        (isinstance(eigenvalues, list) and 1<=len(eigenvalues)<=2 and isinstance(eigenvalues[0], torch.Tensor)):
        full_eigenvalues: torch.Tensor = from_kron_block_decomposed_to_tensor([eigenvalues])
        top_eigenvalue_indices: List[int] = pca_mask(full_eigenvalues, coverage)
        # Compute the indices of the Kronecker product
        if len(eigenvalues) == 2:
            kron1_idxs, kron2_idxs = np.unravel_index(top_eigenvalue_indices, (len(eigenvalues[0]), len(eigenvalues[1])))
            return list(zip(kron1_idxs, kron2_idxs))
        else:
            return top_eigenvalue_indices
    elif isinstance(eigenvalues, list):
        top_eigenvalue_indices = []
        for eigenvalue_decomposed in eigenvalues:
            top_eigenvalue_indices.append(pca_mask(eigenvalue_decomposed, coverage))
    else:
        raise ValueError("Either top or coverage must be specified")

    return top_eigenvalue_indices


def log_laplace_eigenvalues(laplace_curv: laplace.ParametricLaplace, logger: TensorBoardLogger):
    weight_eigenvalues, weight_eigenvectors = compute_laplace_eigendecomp(laplace_curv)

    if isinstance(laplace_curv, laplace.FullLaplace):
        prefix: Text = "laplace_eig/full"
        layer_names = None
    elif isinstance(laplace_curv, laplace.KronLaplace):
        prefix: Text = "laplace_eig/kron"
        model_decompostion = compute_model_decomposition(laplace_curv.model)
        layer_names = list(model_decompostion.keys())
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")

    fig_prefix = f"{prefix}/hist"
    for (layer_name, fig) in histogram_eigenvalues(weight_eigenvalues, layer_names=layer_names).items():
        fig_name = f"{fig_prefix}/{layer_name}" if layer_name != '' else fig_prefix
        logger.experiment.add_figure(fig_name, fig)

    
def log_laplace_loadings(laplace_curv: laplace.ParametricLaplace, logger: TensorBoardLogger, params: List[Tuple[NetworkGranularity, Optional[ReduceMode]]]):
    weight_eigenvalues, weight_eigenvectors = compute_laplace_eigendecomp(laplace_curv)
    model_decompostion = compute_model_decomposition(laplace_curv.model)

    # print(f"Finding the most important principal components(eigenvectors) of the Laplace approximation")
    top_eigenvector_indices = pca_mask(weight_eigenvalues, coverage=0.9)
    # print(f"From {len(weight_eigenvalues)} eigenvalues and eigenvectors, {len(top_eigenvector_indices)} cover 95% of the total sum of eigenvalues.")
    top_eigenvalues = mask_pca_eigenvalues(weight_eigenvalues, top_eigenvector_indices)

    # for (i, (eigenvalue_tensor, top_eigenvalue_tensor)) in enumerate(zip(from_kron_block_decomposed_to_tensors(weight_eigenvalues), from_kron_block_decomposed_to_tensors(top_eigenvalues))):
    #     print(f"Selection of top eigenvalues for layer {i}: shape: {eigenvalue_tensor.shape} -> {top_eigenvalue_tensor.shape}")
        
    top_eigenvectors = mask_pca_eigenvectors(weight_eigenvectors, top_eigenvector_indices)

    # for (i, (eigenvector_tensor, top_eigenvector_tensor)) in enumerate(zip(from_kron_block_decomposed_to_tensors(weight_eigenvectors), from_kron_block_decomposed_to_tensors(top_eigenvectors))):
    #     print(f"Selection of top PCs for layer {i}: shape: {eigenvector_tensor.shape} -> {top_eigenvector_tensor.shape}")

    if isinstance(laplace_curv, laplace.FullLaplace):
        prefix: Text = "laplace_loading/full"
    elif isinstance(laplace_curv, laplace.KronLaplace):
        prefix: Text = "laplace_loading/kron"
    else:
        raise ValueError(f"Unknown type of LaPlace approximation {type(laplace_curv)}")

    # pca_loadings = compute_pca_loadings(weight_eigenvalues, weight_eigenvectors)
    top_pca_loadings = compute_pca_loadings(top_eigenvalues, top_eigenvectors)
    # for (i, (loading_tensor, top_loading_tensor)) in enumerate(zip(from_kron_block_decomposed_to_tensors(pca_loadings), from_kron_block_decomposed_to_tensors(top_pca_loadings))):
    #     print(f"Selection of top loadings for layer {i}: shape: {loading_tensor.shape} -> {top_loading_tensor.shape}")
    
    for (x_axis, reduce_mode) in params:
        fig_prefix = f"{prefix}/pca_loadings/{x_axis}/{reduce_mode}"
        for (fig_title, heatmap_figure) in heatmap_loadings(top_pca_loadings, model_decomposition=model_decompostion, reduce_mode=reduce_mode, x_axis=x_axis):
            # print(f"Logging figure {fig_prefix}/{fig_title}")
            fig_name = f"{fig_prefix}/{fig_title}" if fig_title != '' else fig_prefix
            logger.experiment.add_figure(fig_name, heatmap_figure)


def compute_pca_loadings(eigenvalues, eigenvectors) -> Union[torch.Tensor, KronBlockDecomposed]:
    # eigenvalues, eigenvectors = compute_laplace_eigendecomp(laplace_curv)
    loadings: Union[torch.Tensor, KronBlockDecomposed]
    if isinstance(eigenvalues, torch.Tensor) and isinstance(eigenvectors, torch.Tensor):
        loadings = torch.matmul(eigenvectors, torch.diag(torch.sqrt(eigenvalues)))
    # If eigenvalues and eigenvectors are KronBlockDecomposed
    elif isinstance(eigenvalues, list) and isinstance(eigenvectors, list):
        loadings = []
        for (eigenvalue_decomposed, eigenvector_decomposed) in zip(eigenvalues, eigenvectors):
            loading = []
            for (eigenvalue, eigenvector) in zip(eigenvalue_decomposed, eigenvector_decomposed):
                assertion.assert_equals(eigenvector.size(1), eigenvalue.size(0))
                loading.append(torch.matmul(eigenvector, torch.diag(torch.sqrt(eigenvalue))))
            loading = tuple(loading)
            loadings.append(loading)

            # assertion.assert_tensor_close(from_kron_block_decomposed_to_tensor([loading]),
            #                               torch.matmul(from_kron_block_decomposed_to_tensor([eigenvector_decomposed]), torch.diag(torch.sqrt(from_kron_block_decomposed_to_tensor([eigenvalue_decomposed])))))

    else:
        raise ValueError(f"Unknown type of eigenvalues {type(eigenvalues)} and eigenvectors {type(eigenvectors)}")
    
    return loadings



def heatmap_loadings(loadings: Union[torch.Tensor, KronBlockDecomposed], x_axis: NetworkGranularity= 'layer_channel', reduce_mode: Optional[ReduceMode] = None, model_decomposition: Optional[Dict[Text, List[int]]]=None, threshold: Optional[Union[torch.Tensor, float]] = 0.2) -> Iterable[Tuple[Text, figure.Figure]]:
    # if isinstance(loadings, list):
    #     loadings_shape = [[list(loading.shape) for loading in layer_loadings] for layer_loadings in loadings]
    # else:
    #     loadings_shape = [list(loadings.shape)]
    # print(f"Preparing heatmap for loadings of shape {loadings_shape}")

    threshold_value: Optional[torch.Tensor] = None

    if isinstance(loadings, list):
        vmin = kron_block_decomposed_min(loadings)
        vmax = kron_block_decomposed_max(loadings)

        if threshold is not None:
            threshold_value = torch.tensor([vmin, vmax]).abs().max() * threshold if isinstance(threshold, float) else threshold

        for (layer_name, layer_loadings) in zip(model_decomposition.keys(), from_kron_block_decomposed_to_tensors(loadings)):
            # print(f"\tPreparing heatmap for layer {layer_name} with loadings of shape {layer_loadings.shape}")
            fig, ax = plt.subplots(figsize=(10,10))
            layer_loadings = layer_loadings.detach()

            if x_axis == "weight":
                # print(f"\tAsserting that all values are between {vmin} and {vmax}")
                # assertion.assert_le(vmin, np.nanmin(data))
                # assertion.assert_le(np.nanmax(data), vmax)
                # Cumsum the number of parameters per channel
                # print(f"\tCumsumming the number of parameters per channel to identify channel indices")
                channel_cumsum = np.cumsum([0] + model_decomposition[layer_name])
                labels = {channel_cumsum[i]: f'{layer_name}.{i}' for i in range(len(channel_cumsum))}

                # print(f"\tPassing data to heatmap of shape {data.shape} with labels {labels} to be plotted")
                if threshold_value is not None:
                    layer_loadings[layer_loadings.abs() <= threshold] = torch.nan
                plot_util.heatmap(
                    data=layer_loadings,
                    row_labels=labels,
                    col_labels=[], 
                    vmin=vmin, vmax=vmax,
                    ax=ax)   
                # print(f"\tYielding heatmap for layer {layer_name}")
                yield layer_name, fig

            elif x_axis == 'layer_channel':
                verification.check_not_none(reduce_mode)
                verification.check_not_none(model_decomposition)
                layer_channel_eigenvalues = reduce_loadings_per_layer_channel(layer_loadings, {layer_name: model_decomposition[layer_name]}, reduce_mode)
                result = {}
                for (layer_name, channel_eigenvalues) in layer_channel_eigenvalues.items():
                    fig, ax = plt.subplots(figsize=(10,10))
                    if threshold_value is not None:
                        # print(f"Setting threshold to value {threshold_value}")
                        channel_eigenvalues[channel_eigenvalues.abs() <= threshold_value] = torch.nan
                    labels = {i: f'{layer_name}.{i}' for i in range(len(channel_eigenvalues)) if (i%10==0 or i==len(channel_eigenvalues)-1)}
                    plot_util.heatmap(
                        data=channel_eigenvalues,
                        row_labels=labels,
                        ax=ax,
                        aspect=channel_eigenvalues.shape[1] / channel_eigenvalues.shape[0], # To make the plot look square
                        vmin=vmin, vmax=vmax)   
                    yield f'{layer_name}', fig
            else:
                raise NotImplementedError(f"Unknown x_axis {x_axis}")
            # print(f"\tAsserting that all values are between {vmin} and {vmax}")
            # assertion.assert_le(vmin, np.nanmin(data))
            # assertion.assert_le(np.nanmax(data), vmax)
            # Cumsum the number of parameters per channel
            # print(f"\tCumsumming the number of parameters per channel to identify channel indices")
    # WARN: untested part
    # elif isinstance(loadings, torch.Tensor):
    #     fig, ax = plt.subplots(figsize=(10,10))
    #     data = loadings.detach().cpu().numpy()
    #     if len(data.shape) == 1:
    #         data = np.diag(data)
    #     data = np.where(data == 0, np.nan, data)
    #     labels = []
    #     plot_util.heatmap(
    #         data=data,
    #         row_labels=labels,
    #         col_labels=labels, 
    #         ax=ax)   
    #     yield '', fig
    else:
        raise ValueError(f"Unknown type of loadings {type(loadings)}")


def histogram_eigenvalues(weight_eigenvalues: Union[torch.Tensor, KronBlockDecomposed], layer_names: Optional[List[Text]] = None) -> Dict[Text, figure.Figure]:
    if isinstance(weight_eigenvalues, torch.Tensor):
        data = weight_eigenvalues.detach().cpu().numpy() 
        fig, ax = plt.subplots()
        ax.hist(data)
        return {'': fig}
    elif isinstance(weight_eigenvalues, list):
        result = {}
        verification.check_not_none(layer_names)
        all_eigenvalues: List[torch.Tensor] = []
        for (layer_name, layer_eigenvalues) in zip(layer_names, from_kron_block_decomposed_to_tensors(weight_eigenvalues)):
            fig, ax = plt.subplots()
            ax.set_yscale('log')
            ax.hist(layer_eigenvalues.detach().cpu().numpy())
            all_eigenvalues.append(layer_eigenvalues)
            result[f'histogram/{layer_name}'] = fig
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        # data = torch.stack(all_eigenvalues).detach().cpu().numpy()
        # print(f"Plotting histogram of shape {data.shape} with {len(layer_names)} layers: {layer_names}")
        ax.hist(all_eigenvalues, stacked=True, label=layer_names)
        ax.legend()
        result[f'histogram/all'] = fig
    return result

