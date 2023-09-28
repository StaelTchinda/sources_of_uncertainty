from typing import Any, Dict, Text, Optional
import pytorch_lightning as pl
from torch.utils import data as torch_data

from util import assertion


class CustomizableDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_dataset_params: Dict[Text, Any],
                 val_dataset_params: Dict[Text, Any],
                 test_dataset_params: Dict[Text, Any],
                 train_dataloader_params: Dict[Text, Any],
                 val_dataloader_params: Dict[Text, Any],
                 test_dataloader_params: Dict[Text, Any]) -> None:
        self.train_dataset_params = train_dataset_params
        self.val_dataset_params = val_dataset_params
        self.test_dataset_params = test_dataset_params
        self.train_dataloader_params = train_dataloader_params
        self.val_dataloader_params = val_dataloader_params
        self.test_dataloader_params = test_dataloader_params

        self.train_dataset: Optional[torch_data.Dataset] = None
        self.val_dataset: Optional[torch_data.Dataset] = None
        self.test_dataset: Optional[torch_data.Dataset] = None
        super().__init__()

    def check_setup(self, stage=None):
        if stage == "fit" or stage is None:
            assertion.assert_not_none(self.train_dataset)
            assertion.assert_not_none(self.val_dataset)
        if stage == "validate" or stage is None:
            assertion.assert_not_none(self.val_dataset)
        if stage == "test" or stage is None:
            assertion.assert_not_none(self.test_dataset)

    def prepare_data(self):
        raise NotImplementedError()
    
    def setup(self, stage=None):
        raise NotImplementedError()

    def train_dataloader(self):
        return torch_data.DataLoader(self.train_dataset, **self.train_dataloader_params)

    def val_dataloader(self):
        return torch_data.DataLoader(self.val_dataset, **self.val_dataloader_params)

    def test_dataloader(self):
        return torch_data.DataLoader(self.test_dataset, **self.test_dataloader_params)

