# From https://gist.github.com/martinferianc/db7615c85d5a3a71242b4916ea6a14a2

import os
import urllib.request
from os import path
from pathlib import Path
from typing import Dict, Text, Union

import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset


class UCIDataset(Dataset):
    uci_datasets = {
        "housing": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            "num_classes": 2
        },
        "concrete": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        },
        "energy": {
            "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        },
        "wine": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "num_classes": 6,
            "read_kwargs": {
                "sep": ";"
            }
        },
        "iris": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
            "num_classes": 4
        },
        "adult": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "num_classes": 2
        },
        "car": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
            "num_classes": 4
        },
        "image": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test",
            "num_classes": 7,
            "label_column": "LABEL",
            "read_kwargs": {
                "header": 2,
                "names": ["LABEL", "REGION-CENTROID-COL", "REGION-CENTROID-ROW", "REGION-PIXEL-COUNT",
                          "SHORT-LINE-DENSITY-5", "SHORT-LINE-DENSITY-2", "VEDGE-MEAN", "VEDGE-SD", "HEDGE-MEAN",
                          "HEDGE-SD", "INTENSITY-MEAN", "RAWRED-MEAN", "RAWBLUE-MEAN", "RAWGREEN-MEAN", "EXRED-MEAN",
                          "EXBLUE-MEAN", "EXGREEN-MEAN", "VALUE-MEAN", "SATURATION-MEAN", "HUE-MEAN"]

            }
        }
    }

    def __init__(self, name: Text, data_path: Union[Path, Text] = ""):
        self._load_uci_dataset(name, data_path)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.Y[idx]

    def _load_uci_dataset(self, name: Text, data_path: Union[Path, Text] = ""):
        if name not in UCIDataset.uci_datasets:
            raise Exception(f"Not known dataset {name}!")
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if not path.exists(data_path / "UCI"):
            os.mkdir(data_path / "UCI")

        url: Text = UCIDataset.uci_datasets[name]["url"]
        file_name: Text = url.split('/')[-1]
        if not path.exists(data_path / "UCI" / file_name):
            urllib.request.urlretrieve(
                UCIDataset.uci_datasets[name]["url"], data_path / "UCI" / file_name)

        file_path: Path = data_path / 'UCI' / file_name
        read_kwargs = UCIDataset.uci_datasets[name]["read_kwargs"] if "read_kwargs" in UCIDataset.uci_datasets[
            name] else {}
        original_df = UCIDataset._read_data(file_path, **read_kwargs)
        self.original_df = original_df
        df = original_df.copy()

        label_column = UCIDataset.uci_datasets[name]["label_column"] if "label_column" in UCIDataset.uci_datasets[
            name] else df.columns[-1]

        self.label_encoders: Dict[pd.Index, preprocessing.LabelEncoder] = {}
        for (column_idx, dtype) in enumerate(df.dtypes):
            column = df.columns[column_idx]
            if str(dtype) == "object":
                label_encoder = preprocessing.LabelEncoder()
                label_encoder.fit(df[column])
                df[column] = label_encoder.transform(df[column])
                self.label_encoders[column] = label_encoder

        self.transformed_df = df
        X = df.loc[:, df.columns != label_column].values
        self.X = torch.tensor(X, dtype=torch.float32)
        Y = df[label_column].values
        self.Y = torch.tensor(Y, dtype=torch.int64)

    @staticmethod
    def _read_data(file_path: Path, **kwargs) -> pd.DataFrame:
        excel_extensions = ['xslx', 'xsl']
        csv_extensions = ['data', 'csv', 'test']

        if not file_path.exists():
            raise ValueError(f"The file {file_path} does not exist.")

        if any(str(file_path).endswith(f'.{ext}') for ext in excel_extensions):
            df = pd.read_excel(file_path, header=0, **kwargs)
        elif any(str(file_path).endswith(f'.{ext}') for ext in csv_extensions):
            df = pd.read_csv(file_path, **kwargs)
        else:
            raise ValueError(f"Data file {file_path.name} has unexpected type.")

        return df
