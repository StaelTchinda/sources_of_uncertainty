import os
from typing import Callable, Optional

import numpy as np
import torch.utils.data.dataset as dataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

CORRUPTIONS_3DCC = ('near_focus', 'far_focus', 'bit_error', 'color_quant', 
                    'flash', 'fog_3d', 'h265_abr', 'h265_crf', 'iso_noise', 
                    'low_light', 'xy_motion_blur', 'z_motion_blur')


# Class inspired from https://gist.github.com/edadaltocg/a5a3bf4175ff129a3c1091286a24e91b 
class CIFAR10_C(dataset.Dataset):
    base_folder = "CIFAR-10-C"
    filename = "CIFAR-10-C.tar"
    file_md5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"

    def __init__(
        self,
        root: str,
        corruption: str,
        intensity: int,
        transform: Optional[Callable] = None,
        download: bool = False,
        N: int = 10000
    ) -> None:
        self.N = N
        self.root = os.path.expanduser(root)
        self.corruption = corruption
        self.intensity = intensity
        self.transforms = transform
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )
        indices = self._indices
        images = np.load(self._perturbation_array)[indices]
        labels = np.load(self._labels_array)[indices]
        self.arrays = [images, labels]

    def __getitem__(self, index):
        x = self.arrays[0][index]

        if self.transforms:
            x = self.transforms(x)

        y = self.arrays[1][index]
        return x, y

    def __len__(self) -> int:
        return len(self.arrays[0])

    @property
    def _indices(self):
        N = self.N
        return slice((self.intensity - 1) * N, self.intensity * N)

    @property
    def _dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def _perturbation_array(self):
        return os.path.join(self._dataset_folder, self.corruption + ".npy")

    @property
    def _labels_array(self):
        return os.path.join(self._dataset_folder, "labels.npy")

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.file_md5
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def _check_exists(self) -> bool:
        return os.path.exists(self._perturbation_array)

    def download(self) -> None:
        if self._check_integrity() and self._check_exists():
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root,
            remove_finished=True,
            md5=self.file_md5,
        )


# Official repository: `https://github.com/hendrycks/robustness`
