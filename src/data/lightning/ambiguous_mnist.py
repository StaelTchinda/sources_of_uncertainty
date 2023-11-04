from typing import Any, Dict, List, Text
from torch.utils import data as torch_data
from torchvision import datasets
import ddu_dirty_mnist

from data.lightning.datamodule import CustomizableDataModule
from data.lightning.util import adapt_dataset_size, random_indices_split



class AmbiguousMnistDataModule(CustomizableDataModule):
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
        ddu_dirty_mnist.AmbiguousMNIST(train=True, download=True, **self.train_dataset_params)
        ddu_dirty_mnist.AmbiguousMNIST(train=False, download=True, **self.test_dataset_params)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None or stage == "validate":
            fit_mode_full_dataset = ddu_dirty_mnist.AmbiguousMNIST(train=True, **self.train_dataset_params)
            fit_mode_full_dataset = adapt_dataset_size(fit_mode_full_dataset, [self.train_ratio, self.val_ratio])
            train_val_indices_split = random_indices_split(fit_mode_full_dataset, [self.train_ratio, self.val_ratio])

            if stage == "fit" or stage is None:
                self.train_dataset = torch_data.Subset(fit_mode_full_dataset, train_val_indices_split[0])

            val_mode_full_dataset = ddu_dirty_mnist.AmbiguousMNIST(train=True, **self.val_dataset_params)
            self.val_dataset = torch_data.Subset(val_mode_full_dataset, train_val_indices_split[1])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ddu_dirty_mnist.AmbiguousMNIST(train=False, **self.test_dataset_params)
