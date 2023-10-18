from typing import Any, Dict, Text
from torch.utils import data as torch_data
from torchvision import datasets
from data.lightning.datamodule import CustomizableDataModule


class MnistDataModule(CustomizableDataModule):
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
        datasets.MNIST(train=True, download=True, **self.train_dataset_params)
        datasets.MNIST(train=False, download=True, **self.test_dataset_params)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None or stage == "validate":
            mnist_full = datasets.MNIST(train=True, **self.train_dataset_params)
            if isinstance(self.train_ratio, int) and isinstance(self.val_ratio, int):
                if len(mnist_full) > self.train_ratio + self.val_ratio:
                    # Get a subset of the dataset
                    mnist_full = torch_data.Subset(mnist_full, range(self.train_ratio + self.val_ratio))
            self.train_dataset, self.val_dataset = torch_data.random_split(mnist_full, [self.train_ratio, self.val_ratio])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.MNIST(
                train=False, **self.test_dataset_params
            )
