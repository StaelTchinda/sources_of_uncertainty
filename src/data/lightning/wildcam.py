from pathlib import Path
import wilds
from wilds.datasets import wilds_dataset
from wilds.common import data_loaders as wilds_loaders
from typing import Any, Dict, Text, Union
from torch.utils import data as torch_data
from torchvision import datasets
from data.lightning.datamodule import CustomizableDataModule


class WildcamDataModule(CustomizableDataModule):
    def __init__(self, 
                 root: Union[Path, Text], 
                 train_dataset_params: Dict[Text, Any],
                 val_dataset_params: Dict[Text, Any],
                 test_dataset_params: Dict[Text, Any],
                 train_dataloader_params: Dict[Text, Any],
                 val_dataloader_params: Dict[Text, Any],
                 test_dataloader_params: Dict[Text, Any]):

        self.root = root

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
        wilds.get_dataset(dataset="iwildcam", download=True, root_dir=self.root)

    def setup(self, stage=None):
        wildcam_full: wilds_dataset.WILDSDataset = wilds.get_dataset(dataset="iwildcam", download=False, root_dir=self.root)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = wildcam_full.get_subset(**self.train_dataset_params)

        # Assign val datasets for use in dataloaders
        if stage == "fit" or stage is None or stage == "validate":
            self.val_dataset = wildcam_full.get_subset(**self.val_dataset_params)


        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = wildcam_full.get_subset(**self.test_dataset_params)

    def train_dataloader(self):
        return wilds_loaders.get_train_loader("standard", self.train_dataset, **self.train_dataloader_params)

    def test_dataloader(self):
        return wilds_loaders.get_eval_loader("standard", self.test_dataset, **self.train_dataloader_params)
