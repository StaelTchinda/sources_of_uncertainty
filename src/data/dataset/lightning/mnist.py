import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str, transform, train_ratio: int, val_ratio: int):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.batch_size = batch_size
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        print(f"Setup: stage={stage}")
        if stage == "fit" or stage is None or stage == "validate":
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [self.train_ratio, self.val_ratio])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
