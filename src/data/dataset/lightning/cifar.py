from typing import Any, Dict, Text, Optional
import pytorch_lightning as pl
from torch.utils import data as torch_data
from torchvision import datasets, transforms

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



class CIFAR10DataModule(CustomizableDataModule):
    def __init__(self, 
                 train_ratio: int, 
                 val_ratio: int,
                 train_dataset_params: Dict[Text, Any],
                 val_dataset_params: Dict[Text, Any],
                 test_dataset_params: Dict[Text, Any],
                 train_dataloader_params: Dict[Text, Any],
                 val_dataloader_params: Dict[Text, Any],
                 test_dataloader_params: Dict[Text, Any]):

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

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
        datasets.CIFAR10(train=True, download=True, **self.train_dataset_params)
        datasets.CIFAR10(train=False, download=True, **self.test_dataset_params)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None or stage == "validate":
            mnist_full = datasets.CIFAR10(train=True, **self.train_dataset_params)
            self.train_dataset, self.val_dataset = torch_data.random_split(mnist_full, [self.train_ratio, self.val_ratio])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                train=False, **self.test_dataset_params
            )
