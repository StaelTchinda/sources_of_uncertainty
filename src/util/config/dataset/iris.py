from typing import List, Tuple, Union
from torch.utils import data as torch_data


from data.dataset.uci import UCIDataset
from util import verification
from util import data as data_util
from util.config import path as path_config

def get_default_datasets_params():
    return {
        'name': 'iris',
        'data_path': path_config.DATA_PATH,
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
    }

def get_default_dataloaders_params():
    return {
        'train': {
            'shuffle': False,
            'batch_size': 32,
            'num_workers': 4,
        },
        'val': {
            'shuffle': False,
            'batch_size': 30,
            'num_workers': 4,
        },
        'test': {
            'shuffle': False,
            'batch_size': 32,
            'num_workers': 4,
        },
    }

def get_default_laplace_dataloaders_params():
    return {
        'train': {
            'shuffle': False,
            'batch_size': 4,
            'num_workers': 4,
        },
        'val': {
            'shuffle': False,
            'batch_size': 30,
            'num_workers': 4,
        },
        'test': {
            'shuffle': False,
            'batch_size': 32,
            'num_workers': 4,
        },
    }


def get_default_datasets() -> Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]:
    params = get_default_datasets_params()
    whole_dataset = UCIDataset(name=params['name'], data_path=params['data_path'])
    train_dataset, val_dataset, test_dataset = data_util.split_data(whole_dataset, [params['train_ratio'], params['val_ratio'], params['test_ratio']])
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
