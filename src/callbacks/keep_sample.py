from typing import Callable, List, Literal, Text, Tuple, Optional
import warnings
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import torch
import pytorch_lightning as pl


from util import assertion, verification


from typing import Callable

class BatchFilter(Callable):
    def __init__(self, correct: Optional[bool] = None, label: Optional[int] = None):
        self.correct = correct
        self.label = label

    def __call__(self, batch, outputs) -> List[int]:
        if len(outputs.shape) == 2:
            probs = outputs
        elif len(outputs.shape) == 3:
            probs = outputs.mean(dim=1)
        else:
            raise ValueError(f"Output shape {outputs.shape} not supported")

        gt_labels = batch[1]
        pred_labels = probs.argmax(-1)

        mask = torch.ones_like(gt_labels, dtype=torch.bool)

        if self.correct is not None:
            if self.correct:
                mask = torch.logical_and(mask, gt_labels==pred_labels)
            else:
                mask = torch.logical_and(mask, gt_labels!=pred_labels)
        if self.label is not None:
            mask = torch.logical_and(mask, gt_labels==self.label)

        idxs = mask.nonzero().squeeze(-1).tolist()    
        return idxs


class BatchVarianceScorer(Callable):
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
        batch_size = batch[1].size(0)
        if outputs.size(0) == batch_size:
            return outputs.var(0).mean(-1)
        elif outputs.size(1) == batch_size:
            return outputs.var(1).mean(-1)
        else:
            raise ValueError(f"Unexpected outputs shape {outputs.shape}")


class KeepSamplesCallback(pl.Callback):
    samples: torch.Tensor
    outputs: torch.Tensor
    gt_labels: torch.Tensor
    score_values: torch.Tensor

    samples_count: int
    highest: bool
    device: torch.device

    def __init__(self, samples_count: int = 5, highest: bool = True, device: torch.device = torch.device('cpu'), stage: Literal['train', 'val', 'test'] = 'val'):
        self.stage = stage

        self.samples_count = samples_count
        self.highest = highest
        self.device = device

        self.reset()

    
    def filter_inputs(self, batch: Tuple[torch.Tensor], outputs: torch.Tensor) -> List[int]:
        raise NotImplementedError()
    
    def compute_scores(self, batch: Tuple[torch.Tensor], outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()	

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        assertion.assert_le(len(batch), 2, f"Expected batch to contain at least 2 elements (samples and labels), but got size {len(batch)}")
        assertion.assert_equals(len(batch[0]), len(batch[1]), f"Expected batch samples and labels to have the same size, but got {len(batch[0])} and {len(batch[1])}")
        # TODO: update the code to get rid of this fix. The first channel of the output should be the batch size, not the sampling size
        if len(outputs.shape) == 3:
            # Switch the first and second channel
            outputs = outputs.transpose(0, 1)
        assertion.assert_equals(len(batch[0]), len(outputs), f"Expected batch and outputs to have the same size, but got {len(batch[0])} and {len(outputs)}")

        relevant_idxs = self.filter_inputs(batch, outputs)
        if len(relevant_idxs) == 0:
            return
        batch = tuple([tensor[relevant_idxs] for tensor in batch])
        outputs = outputs[relevant_idxs]

        samples: torch.Tensor = batch[0]
        gt_labels: torch.Tensor = batch[1]

        score_values = self.compute_scores(batch, outputs)
        top_score_values, top_score_indices = torch.topk(score_values, k=min(len(score_values), self.samples_count), largest=self.highest)

        assertion.assert_tensor_close(top_score_values, torch.sort(top_score_values, descending=self.highest).values)

        original_score_values = self.score_values.clone()

        new_sorted_score_values_next_index: int = 0
        for index in range(self.samples_count):
            new_score_values_next_index: Optional[int] = None

            if new_sorted_score_values_next_index >= len(top_score_values):
                break

            if index >= len(self.score_values): # if we have not yet saved enough metric values
                new_score_values_next_index = int(top_score_indices[new_sorted_score_values_next_index].item())
                new_sorted_score_values_next_index += 1
            else:
                compare = lambda x, y, highest: x>y if highest else x<y
                if compare(top_score_values[new_sorted_score_values_next_index], self.score_values[index], self.highest):
                    new_score_values_next_index = int(top_score_indices[new_sorted_score_values_next_index].item())

            if new_score_values_next_index is not None:
                self.samples = insert_tensor_in_tensor(self.samples, samples[new_score_values_next_index].to(self.device), index)
                self.outputs = insert_tensor_in_tensor(self.outputs, outputs[new_score_values_next_index].to(self.device), index)
                self.gt_labels = insert_tensor_in_tensor(self.gt_labels, gt_labels[new_score_values_next_index].to(self.device), index)
                self.score_values = insert_tensor_in_tensor(self.score_values, score_values[new_score_values_next_index].to(self.device), index)
        
                assertion.assert_tensor_close(self.score_values, torch.sort(self.score_values, descending=self.highest).values)

                if len(self.score_values) > self.samples_count:
                    self.samples = self.samples[:self.samples_count]
                    self.outputs = self.outputs[:self.samples_count]
                    self.gt_labels = self.gt_labels[:self.samples_count]
                    self.score_values = self.score_values[:self.samples_count]

                self._assert_shape_match()

        if len(original_score_values) + len(top_score_values) > len(self.score_values) and \
            len(self.score_values) < self.samples_count:
            warnings.warn(f"Could not further fill the callback container with {self.samples_count} samples, although got {len(original_score_values)} original scores and {len(top_score_values)} new scores. Only {len(self.score_values)} samples were found.")
            self._assert_shape_match()

    def _assert_shape_match(self):
        assertion.assert_le(self.samples.size(0), self.samples_count)
        assertion.assert_equals(self.samples.size(0), self.outputs.size(0))
        assertion.assert_equals(self.samples.size(0), self.gt_labels.size(0))
        assertion.assert_equals(self.samples.size(0), self.score_values.size(0))

    def reset(self):
        self.samples = torch.tensor([])
        self.outputs = torch.tensor([])
        self.gt_labels = torch.tensor([])
        self.score_values = torch.tensor([])

    @property
    def probs(self):
        if len(self.outputs.shape) == 2:
            return self.outputs
        elif len(self.outputs.shape) == 3:
            return self.outputs.mean(dim=1)
        else: 
            raise ValueError(f"Unexpected outputs to have shape {self.outputs.shape}")

def insert_tensor_in_tensor(container: torch.Tensor, element: torch.Tensor, index: int):
    # print(f"Inserting element of shape {element.shape} into container of shape {container.shape} at index {index}")
    assertion.assert_le(index, len(container))
    return torch.cat([container[:index], element.unsqueeze(0), container[index:]], 0)


class KeepSamplesLambdaCallback(KeepSamplesCallback):
    def __init__(self, filter: Callable, scorer: Callable, **kwargs):
        self._filter_inputs = filter
        self._compute_scores = scorer
        super().__init__(**kwargs)

    def filter_inputs(self, batch: Tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> List[int]:
        idxs = self._filter_inputs(batch, outputs)
        return idxs

    def compute_scores(self, batch: Tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor) -> List[int]:
        return self._compute_scores(batch, outputs)


class KeepImagesCallbackContainer:
    def __init__(self, filters: List[Callable], scorers: List[Callable], **kwargs):
        verification.check_equals(len(filters), len(scorers))
        self.callbacks: List[KeepSamplesLambdaCallback] = []
        for (filter, scorer) in zip(filters, scorers):
            self.callbacks.append(KeepSamplesLambdaCallback(filter=filter, scorer=scorer, **kwargs))

    def plot(self,
             title: Optional[Text] = None,
             figsize=(10,12)) -> Figure:

        # count = sum([len(callback.samples) for callback in self.callbacks])
        ncols = max([len(callback.samples) for callback in self.callbacks])
        nrows = len(self.callbacks)


        if ncols==0 or nrows==0:
            return plt.figure(figsize=figsize)

        fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

        ax_title_description = f"y_true ; y_pred ; prob ; score"
        if title is not None:
            title += f"\n{ax_title_description}"
        else:
            title = ax_title_description

        fig.suptitle(title)

        for i in range(nrows):
            callback = self.callbacks[i]
            for j in range(ncols):
                if ncols > 1:
                    axis = ax[i, j]
                else:
                    axis = ax[i]
                
                axis.set_xticks([], [])
                axis.set_yticks([], [])

                if j >= len(callback.samples):
                    continue
                
                if len(callback.probs.shape) == 3:
                    probs = callback.probs[j].mean(dim=0)
                elif len(callback.probs.shape) == 2:
                    probs = callback.probs[j]
                else:
                    raise ValueError(f"Unexpected outputs to have shape {callback.probs.shape}")
                assertion.assert_equals(1, len(probs.shape))

                pred = probs.argmax(-1)

                axis.set_title(f"{callback.gt_labels[j]:.0f} ; {pred} ; {probs[pred]:.1E} ; {callback.score_values[j].item():.1E}", fontsize=7, pad=0)
                axis.imshow(callback.samples[j].permute(1,2,0), cmap="gray")
                # TODO: adapt for color images

        return fig