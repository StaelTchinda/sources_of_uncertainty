from typing import Tuple, Union

import torch
from torch.utils import data as torch_data
from torchvision import datasets, transforms
import sklearn
import sklearn.model_selection

from util import data as data_util
from util.config import path as path_config

def get_default_datasets_params():
    cifar10_train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomGrayscale(0.2),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.2),
        transforms.RandomRotation(30),
        transforms.RandomAdjustSharpness(0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    return {
        'train_ratio': 40000,
        'val_ratio': 10000,
        'transform': {
            'train': cifar10_train_transform,
            'val': cifar10_test_transform,
            'test': cifar10_test_transform
        },
    }


def get_default_dataloaders_params():
    return {
        'train': {
            'shuffle': False,
            'batch_size': 256,
            'num_workers': 8,
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

    train_ratio: float = params['train_ratio']
    val_ratio: float = params['val_ratio']

    train_transform = params['transform']['train']
    val_transform = params['transform']['val']
    test_transform = params['transform']['test']

    train_data = datasets.CIFAR10(
        root=str(path_config.DATA_PATH),
        train=True,
        download=True,
        transform=train_transform
    )
    val_data = datasets.CIFAR10(
        root=str(path_config.DATA_PATH),
        train=True,
        download=True,
        transform=val_transform
    )

    test_dataset = datasets.CIFAR10(
        root=str(path_config.DATA_PATH),
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_indices, val_indices = sklearn.model_selection.train_test_split(range(len(train_data)), test_size=val_ratio, train_size=train_ratio)

    train_dataset = torch_data.Subset(train_data, train_indices)
    val_dataset = torch_data.Subset(val_data, val_indices)
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