from typing import List, Union
import torch

from typing import Tuple

from network import lightning as lightning


KronBlock = Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]
KronBlockDecomposed = List[KronBlock]

def from_kron_block_decomposed_to_tensor(kron_block_decomposed: KronBlockDecomposed) -> torch.Tensor:
    torch_kron = lambda l: torch.kron(l[0], l[1]) if len(l) == 2 else l[0]
    return torch.cat([torch_kron(kron_block) for kron_block in kron_block_decomposed])