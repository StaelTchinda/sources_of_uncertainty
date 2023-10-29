from typing import Iterable, List, Union
import torch

from typing import Tuple

from network import lightning as lightning
from util import assertion


KronBlock = Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
KronBlockDecomposed = List[KronBlock]

def from_kron_block_decomposed_to_tensor(kron_block_decomposed: KronBlockDecomposed) -> torch.Tensor:
    torch_kron = lambda l: torch.kron(l[0], l[1]) if len(l) == 2 else l[0]
    return torch.cat([torch_kron(kron_block) for kron_block in kron_block_decomposed])


def from_kron_block_decomposed_to_tensors(kron_block_decomposed: KronBlockDecomposed) -> Iterable[torch.Tensor]:
    torch_kron = lambda l: torch.kron(l[0], l[1]) if len(l) == 2 else l[0]
    for kron_block in kron_block_decomposed:
        yield torch_kron(kron_block)


def kron_block_decomposed_min(kron_block_decomposed: KronBlockDecomposed) -> float:
    torch_kron = lambda l: torch.kron(l[0], l[1]) if len(l) == 2 else l[0]
    torch_kron_min = lambda l: kron_min(l[0], l[1]) if len(l) == 2 else torch.min(l[0]).item()
    block_mins = []
    for kron_block in kron_block_decomposed:
        block_min = torch_kron_min(kron_block)
        assertion.assert_float_close(torch.min(torch_kron(kron_block)).item(), block_min)
        block_mins.append(block_min)
    return min([torch_kron_min(kron_block) for kron_block in kron_block_decomposed])


def kron_block_decomposed_max(kron_block_decomposed: KronBlockDecomposed) -> float:
    torch_kron_max = lambda l: torch.max(l[0]) * torch.max(l[1]) if len(l) == 2 else torch.max(l[0])
    return max([torch_kron_max(kron_block).item() for kron_block in kron_block_decomposed])


def kron_min(kron1: torch.Tensor, kron2: torch.Tensor) -> float:
    # Computes the minimun of the kronecker product of kron1 and kron2 without explicitly computing the kronecker product
    kron1_min = torch.min(kron1)
    kron2_min = torch.min(kron2)
    kron1_positive_max = torch.max(kron1[kron1 > 0])
    kron2_positive_max = torch.max(kron2[kron2 > 0])
    # If one minimum is positive, then the minimum of the kronecker product is the product of the two minimums
    if kron1_min.item()>0 and kron2_min.item()>0:
        return kron1_min.item() * kron2_min.item()
    else:
        return torch.min(
            kron1_min * kron2_positive_max,
            kron1_positive_max * kron2_min
        ).item()

def kron_max(kron1: torch.Tensor, kron2: torch.Tensor) -> float:
    return -kron_min(-kron1, kron2)
            
    