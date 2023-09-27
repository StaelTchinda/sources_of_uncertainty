from typing import Tuple, Union

import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms


from util import data as data_util
from util.config import path as path_config

def get_default_datasets_params():
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return {
        'train_ratio': 50000,
        'val_ratio': 10000,
        'test_ratio': 10000,
        'transform': {
            'train': mnist_transform,
            'test': mnist_transform
        },
    }


def get_default_dataloaders_params():
    return {
        'train': {
            'shuffle': True,
            'batch_size': 32,
            'num_workers': 4,
        },
        'val': {
            'shuffle': False,
            'batch_size': 256,
            'num_workers': 4,
        },
        'test': {
            'shuffle': False,
            'batch_size': 1024,
            'num_workers': 4,
        },
    }

def get_default_laplace_dataloaders_params():
    return {
        'train': {
            'shuffle': False,
            'batch_size': 8,
            'num_workers': 4,
        },
        'val': {
            'shuffle': False,
            'batch_size': 8,
            'num_workers': 4,
        },
        'test': {
            'shuffle': False,
            'batch_size': 32,
            'num_workers': 1024,
        },
    }


def get_default_datasets() -> Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]:
    params = get_default_datasets_params()

    train_ratio: Union[float, int] = params['train_ratio']
    val_ratio: Union[float, int] = params['val_ratio']

    train_transform = params['transform']['test']
    test_transform = params['transform']['train']

    training_data = datasets.MNIST(
        root=str(path_config.DATA_PATH),
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset, val_dataset = data_util.split_data(training_data, [train_ratio, val_ratio])

    # Download test data from open datasets.
    test_dataset = datasets.MNIST(
        root=str(path_config.DATA_PATH),
        train=False,
        download=True,
        transform=test_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_default_dataloaders():
    train_dataset, val_dataset, test_dataset = get_default_datasets()
    params = get_default_dataloaders_params()
    train_dataloader = torch_data.DataLoader(train_dataset, **params['train'])
    val_dataloader = torch_data.DataLoader(val_dataset, **params['val'])
    test_dataloader = torch_data.DataLoader(test_dataset, **params['test'])
    return train_dataloader, val_dataloader, test_dataloader



def get_default_laplace_dataloaders():
    train_dataset, val_dataset, test_dataset = get_default_datasets()
    params = get_default_laplace_dataloaders_params()
    train_dataloader = torch_data.DataLoader(train_dataset, **params['train'])
    val_dataloader = torch_data.DataLoader(val_dataset, **params['val'])
    test_dataloader = torch_data.DataLoader(test_dataset, **params['test'])
    return train_dataloader, val_dataloader, test_dataloader