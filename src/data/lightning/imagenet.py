from typing import Any, Dict, List, Text
from cycler import V
from torch.utils import data as torch_data
from torchvision import datasets
from data.lightning.datamodule import CustomizableDataModule
from data.lightning.util import adapt_dataset_size, random_indices_split


class ImageNetDataModule(CustomizableDataModule):
    def __init__(self, 
                 train_ratio: int, 
                 val_ratio: int,
                 train_dataset_params: Dict[Text, Any],
                 val_dataset_params: Dict[Text, Any],
                 train_dataloader_params: Dict[Text, Any],
                 val_dataloader_params: Dict[Text, Any]):

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        super().__init__(
            train_dataset_params=train_dataset_params,
            val_dataset_params=val_dataset_params,
            test_dataset_params={},
            train_dataloader_params=train_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            test_dataloader_params={},
        )

    def prepare_data(self):
        # download
        datasets.ImageNet(split="train", **self.train_dataset_params)
        datasets.ImageNet(split="val", **self.val_dataset_params)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None or stage == "validate":
            train_dataset = datasets.ImageNet(split="train", **self.train_dataset_params)
            if self.train_ratio:
                train_dataset = adapt_dataset_size(train_dataset, [self.train_ratio])
            self.train_dataset = train_dataset

        if stage == "fit" or stage is None or stage == "validate":
            val_dataset = datasets.ImageNet(split="val", **self.val_dataset_params)
            if self.val_ratio:
                val_dataset = adapt_dataset_size(val_dataset, [self.val_ratio])
            self.val_dataset = val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            raise ValueError("Test dataset is not available for ImageNet dataset")

    def test_dataloader(self):
        raise ValueError("Test dataset is not available for ImageNet dataset")