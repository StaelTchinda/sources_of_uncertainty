from typing import Any, Dict, Text, Union
import warnings
from torch.utils import data as torch_data
from torchvision import datasets
from data.lightning.datamodule import CustomizableDataModule
from data.dataset.uci import UCIDataset


class UciDataModule(CustomizableDataModule):
    def __init__(self, 
                 train_ratio: Union[int, float], 
                 val_ratio: Union[int, float],
                 test_ratio: Union[int, float],
                 train_dataset_params: Dict[Text, Any],
                 val_dataset_params: Dict[Text, Any],
                 test_dataset_params: Dict[Text, Any],
                 train_dataloader_params: Dict[Text, Any],
                 val_dataloader_params: Dict[Text, Any],
                 test_dataloader_params: Dict[Text, Any]):

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if val_dataset_params != {}:
            warnings.warn("val_dataset_params is not empty. This will be ignored.")
        if test_dataset_params != {}:
            warnings.warn("test_dataset_params is not empty. This will be ignored.")
        if val_dataloader_params != {}:
            warnings.warn("val_dataloader_params is not empty. This will be ignored.")

        super().__init__(
            train_dataset_params=train_dataset_params,
            val_dataset_params=val_dataset_params,
            test_dataset_params=test_dataset_params,
            train_dataloader_params=train_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            test_dataloader_params=test_dataloader_params
        )

    def prepare_data(self):
        # download
        UCIDataset(**self.train_dataset_params)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        dataset_full = UCIDataset(**self.train_dataset_params)
        self.train_dataset, self.val_dataset, self.test_dataset = torch_data.random_split(dataset_full, [self.train_ratio, self.val_ratio, self.test_ratio])