


from typing import Optional, Text, Union, Tuple
from pathlib import Path

from torch import nn
import pytorch_lightning as pl
import config
from config.mode import ModelMode
from util import checkpoint

def get_best_checkpoint(model_mode: ModelMode, checkpoints_path: Path, with_path: bool = False) -> Union[Tuple[Optional[pl.LightningModule], Optional[Path]], Optional[pl.LightningModule]]:
    best_checkpoint_path: Optional[Path] = None
    if model_mode=="resnet20":
        best_checkpoint_path = Path("/home/stud/tchindak/storage/tchindak/sources_of_uncertainty/checkpoints/cifar10/resnet20/model/lightning_logs/version_1116921/checkpoints/epoch=124-step=39125-val_loss=1.66-best.ckpt")
    else:
        best_checkpoint_path = checkpoint.find_best_checkpoint(checkpoints_path)
    
    pl_module: Optional[pl.LightningModule] = None
    if best_checkpoint_path is not None:
        model = config.network.get_default_model(model_mode)
        pl_module = config.network.lightning.get_default_lightning_module(model_mode, model)
        pl_module = pl_module.__class__.load_from_checkpoint(str(best_checkpoint_path), model=model)

    if with_path:
        return pl_module, best_checkpoint_path
    else:
        return pl_module


def get_pretrained(model_mode: ModelMode, checkpoints_path: Path, with_path: bool = False) -> Union[Tuple[Optional[pl.LightningModule], Optional[Path]], Optional[pl.LightningModule]]:
    pretrained_path: Optional[Path] = None
    pretrained_path = checkpoint.find_pretrained(checkpoints_path)
    
    pretrained_model: Optional[nn.Module] = None
    if pretrained_path is not None:
        model = config.network.get_default_model(model_mode)
        pretrained_model = checkpoint.load_model(model, file_name=pretrained_path.stem, 
                            path_args={'save_path': pretrained_path.parent, 'file_ext': pretrained_path.suffix[1:]})

    if with_path:
        return pretrained_model, pretrained_path
    else:
        return pretrained_model