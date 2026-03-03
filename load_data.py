"""
Simple script to attempt to load the dataset used to train tarflow
"""
import os
from typing import Callable, Optional, Dict

import torch
from torch import Tensor
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor


import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset, ImageFolder
import torchvision.transforms as T
from pathlib import Path


from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader



import transformer_flow
import torchvision as tv
import torch
import ipdb
from pathlib import Path

if __name__ == "__main__":


    img_size: int = 256
    dataset: str = 'afhq'
    # model = transformer_flow.Model(
    #     in_channels=3,
    #     img_size=256,
    #     patch_size=8,
    #     channels=768,
    #     num_blocks=8,
    #     layers_per_block=8,
    #     nvp=True,
    #     num_classes=3,
    # ).to('cuda')
    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize(img_size),
            tv.transforms.CenterCrop(img_size),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    data: tv.datasets.ImageFolder = tv.datasets.ImageFolder(str("/data/davidn/PhXD/GeometricDeepLearning/RiemannEBM_PLightning/afhq_download_repo/data/afhq"), transform=transform)
    
    class AFHQVisionDataset(VisionDataset):
        def __init__(
            self,
            root: str | Path,
            transform: Optional[Callable] = None,
        ) -> None:
            super().__init__(root, transform=transform)

            # Use ImageFolder strictly for indexing consistency
            self._imagefolder = ImageFolder(root=root, transform=None)

            # Store samples and targets exactly as ImageFolder defines them
            self.samples = self._imagefolder.samples
            self.targets = self._imagefolder.targets
            self.classes = self._imagefolder.classes
            self.class_to_idx = self._imagefolder.class_to_idx

            self.loader = default_loader

            # Public accessor dictionary
            self.data_train: Dict[str, Any] = {
                "img": _LazyImageTensorAccessor(self),
                "label": torch.tensor(self.targets, dtype=torch.long),
            }

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Tensor, int]:
            path, target = self.samples[index]
            img = self.loader(path)

            if self.transform is not None:
                img = self.transform(img)

            return img, target


    class _LazyImageTensorAccessor:
        """
        Allows syntax:

            dset.data_train['img'][mask]

        without preloading images into memory.
        """

        def __init__(self, dataset: AFHQVisionDataset) -> None:
            self.dataset = dataset

        def __getitem__(self, idx) -> Tensor:
            if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                indices = idx.nonzero(as_tuple=False).flatten().tolist()
            elif isinstance(idx, torch.Tensor):
                indices = idx.tolist()
            elif isinstance(idx, slice):
                indices = list(range(*idx.indices(len(self.dataset))))
            elif isinstance(idx, int):
                img, _ = self.dataset[idx]
                return img
            else:
                indices = list(idx)

            imgs = [self.dataset[i][0] for i in indices]
            return torch.stack(imgs, dim=0)
        
    dset = AFHQVisionDataset(root="/data/davidn/PhXD/GeometricDeepLearning/RiemannEBM_PLightning/afhq_download_repo/data/afhq/train", transform=transform)

    ipdb.set_trace()

    print(f"{dset.data_train['img'][0].shape=}")
