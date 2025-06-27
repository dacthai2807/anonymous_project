import os
import torch
from monai.data import Dataset
import glob
from typing import List, Optional, Dict
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    RandSpatialCrop,
    RandAxisFlip,
    RandRotate90,
    ScaleIntensityRange,
    ToTensor,
    AddChannel,
    Lambda
)
# from monai.transforms.utility.array import AddChannel

import json
import numpy as np
# Dùng Lambda để load npy
load_npy = Lambda(lambda x: np.load(x))

from monai.data import CacheDataset
from PIL import Image
# def load_with_augment(image_path: str, augment: callable = None):

class MockMonaiDataset(Dataset):
    def __init__(
        self,
        length: int,
        transform: Compose,
        image_H: int = 512,
        image_W: int = 512,
        image_D: int = 114,
        image_key: str = 'images',
        hint_key: str = 'hint',
        txt_key: str = 'txt'
    ):
        self.length = length
        self.transform = transform
        self.H = image_H
        self.W = image_W
        self.D = image_D
        self.image_key = image_key
        self.hint_key = hint_key
        self.txt_key = txt_key

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Tạo ảnh giả
        image = torch.randint(0, 256, size=(self.D, self.H, self.W), dtype=torch.uint8).numpy()
        caption = f"This is a mock caption for sample {index}"

        # Áp dụng transform (transform expects numpy array)
        transformed_image = self.transform(image)

        return {
            self.image_key: transformed_image,
            self.hint_key: caption,
        }

class MonaiDataModule(LightningDataModule):
    def __init__(
        self,
        image_H: int = 512,
        image_W: int = 512,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        num_train_samples: int = 1000,
        num_val_samples: int = 200,
    ):
        super().__init__()
        self.image_H = image_H
        self.image_W = image_W
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples

        self.train_transforms = Compose([
            AddChannel(),
            # EnsureChannelFirst(),
            RandSpatialCrop(roi_size=(self.image_H, self.image_W), random_size=False),
            RandAxisFlip(prob=0.75),
            RandRotate90(prob=0.75),
            ScaleIntensityRange(clip=True, a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor()
        ])
        
        self.val_transforms = Compose([
            AddChannel(),
            # EnsureChannelFirst(),
            RandSpatialCrop(roi_size=(self.image_H, self.image_W), random_size=False),
            ScaleIntensityRange(clip=True, a_min=0, a_max=255, b_min=0.0, b_max=1.0),
            ToTensor()
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self._train_ds = MockMonaiDataset(
                length=self.num_train_samples,
                transform=self.train_transforms,
                image_H=self.image_H,
                image_W=self.image_W,
                image_key='image',
                hint_key='hint'
            )
            self._val_ds = MockMonaiDataset(
                length=self.num_val_samples,
                transform=self.val_transforms,
                image_H=self.image_H,
                image_W=self.image_W,
                image_key='image',
                hint_key='hint'
            )
        if stage == 'test' or stage is None:
            self._test_ds = self._val_ds

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self._val_ds)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self._test_ds)

    
def main():
    print("="*60)
    print("Mock DataModule Verification")
    print("="*60)

    datamodule = MonaiDataModule(
        image_H=512,
        image_W=512,
        micro_batch_size=2,
        global_batch_size=4,
        num_workers=2,
    )

    datamodule.setup('fit')

    print("\n--- Verifying Training Dataloader ---")
    try:
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        print(f"Train dataloader created. Batch keys: {batch.keys()}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}, min={value.min():.2f}, max={value.max():.2f}")
            else:
                print(f"  - {key}: {type(value)}, sample: {value[0]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
        
        

        