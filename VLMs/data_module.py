import os
import torch
from monai.data import Dataset
import glob
from typing import List, Optional, Dict
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
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
import json
import numpy as np
# Dùng Lambda để load npy
load_npy = Lambda(lambda x: np.load(x))

from monai.data import CacheDataset
from PIL import Image
# def load_with_augment(image_path: str, augment: callable = None):

class MockMonaiDataset(CacheDataset):
    def __init__(self, 
                transform: Compose,
                 image_key: str = "image",
                 hint_key: str = "hint"):
        self.transform = transform
        self.image_key = image_key
        self.hint_key = hint_key

        self.data 
class MonaiDataset(CacheDataset):
    def __init__(self, data: List[Dict], transform: Compose, image_key: str = 'images', hint_key: str = 'hint'):
        self.data = data
        self.transform = transform
        self.image_key =  image_key
        self.hint_key = hint_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        print(sample)
        conv_content = sample["conversations"]
        transformed_item = self.transform(sample["image"])

        final_sample = {
            self.image_key : transformed_item,
            self.hint_key : conv_content
        }


class MonaiDataModule(LightningDataModule):
    def __init__(
        self,
        train_data: str,
        valid_data: str,
        image_H: int = 512,
        image_W: int = 512,
        image_D: int = 114,
        micro_batch_size: int = 1,
        global_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):

        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.image_H = image_H
        self.image_W = image_W
        self.image_D = image_D
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_transforms = Compose([
            load_npy,  # Load từ file .npy
            AddChannel(),
            # EnsureChannelFirst(),
            RandSpatialCrop(roi_size=(self.image_D, self.image_H, self.image_W), random_size=False),
            RandAxisFlip(prob=0.75),
            RandRotate90(prob=0.75),
            ScaleIntensityRange(clip=True, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ToTensor()
        ])
        
        self.val_transforms = Compose([
            load_npy,
            AddChannel(),
            # EnsureChannelFirst(),
            RandSpatialCrop(roi_size=(self.image_D, self.image_H, self.image_W), random_size=False),
            RandAxisFlip(prob=0.0),
            RandRotate90(prob=0.0),
            ScaleIntensityRange(clip=True, a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            ToTensor()
        ])

        self.train_tuples: List[Dict] = []
        self.val_tuples: List[Dict] = []
        self.test_tuples: List[Dict] = []

    def _load_sample(self):
        print(f"Loading image samples from: {self.train_data} and {self.valid_data}")

        # if not os.path.exists(self.train_data) or os.path.exists(self.valid_data):
        #     print(f"Warning: Data directory not found at {self.train_data}")
        #     return

        self.train_tuples = json.load(open(self.train_data, "r"))
        self.val_tuples = json.load(open(self.valid_data, "r"))
        
        print(f"Created {len(self.train_tuples)} training, {len(self.val_tuples)} val")

    def setup(self, stage: Optional[str] = None):
        self._load_sample()
        if stage == 'fit' or stage is None:
            self._train_ds = MonaiDataset(
                self.train_tuples, 
                self.train_transforms, 
                image_key='image', 
                hint_key='hint'
            )
            self._val_ds = MonaiDataset(
                self.val_tuples, 
                self.val_transforms, 
                image_key='image', 
                hint_key='hint', 
            )
        if stage == 'test' or stage is None:
            self._test_ds = MonaiDataset(
                self.val_tuples, 
                self.val_transforms,
                image_key='image', 
                hint_key='hint'
            )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Creates a DataLoader for a given Dataset instance."""
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        return self._create_dataloader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Returns the val DataLoader."""
        return self._create_dataloader(self._val_ds)

    def test_dataloader(self) -> DataLoader:
        """Returns the testing DataLoader."""
        return self._create_dataloader(self._test_ds)
    
def main():
    train_data = "/data/data_remote/PET_report_paired_fixed/pretrain_data/single_turn/align_train.json"
    val_data = "/data/data_remote/PET_report_paired_fixed/pretrain_data/single_turn/align_train.json"
    
    print("="*60)
    print("DataModule Verification")
    print("="*60)

    # Instantiate the datamodule
    datamodule = MonaiDataModule(
        train_data=train_data,
        valid_data = val_data,
        image_H=512,
        image_W=512,
        image_D=114,
        micro_batch_size=2,
        global_batch_size=4,
        num_workers=2,
    )
    # Setup datasets
    datamodule.setup('fit')
    
    # Check if data was loaded
    if not datamodule.train_tuples or not datamodule.val_tuples:
        print("\nWarning: No training or val samples were loaded.")
        print("Please check the 'root_folder' path and the contents of the directory.")
        return
        
    print("\n--- Verifying Training Dataloader ---")
    try:
        train_loader = datamodule.train_dataloader()
        print(f"Train dataloader created with batch size {datamodule.micro_batch_size}.")
        
        # Get one batch to inspect
        batch = next(iter(train_loader))
        
        print("\nSample batch from training data:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}, Min: {value.min():.2f}, Max: {value.max():.2f}")
            elif isinstance(value, list): # For text prompts
                 print(f"  - Key: '{key}', Type: list, Length: {len(value)}, First element: '{value[0][:]}'")
    except Exception as e:
        print(f"An error occurred while testing the train dataloader: {e}")

    print("\n--- Verifying Val Dataloader ---")
    try:
        val_loader = datamodule.val_dataloader()
        print(f"Val dataloader created with batch size {datamodule.micro_batch_size}.")

        batch = next(iter(val_loader))
        
        print("\nSample batch from val data:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - Key: '{key}', Shape: {value.shape}, DType: {value.dtype}, Min: {value.min():.2f}, Max: {value.max():.2f}")
            elif isinstance(value, list):
                 print(f"  - Key: '{key}', Type: list, Length: {len(value)}, First element: '{value[0][:]}'")
    except Exception as e:
        print(f"An error occurred while testing the val dataloader: {e}")
        
    print("\n" + "="*60)
    print("Verification complete.")
    print("If shapes and dtypes look correct, the datamodule is likely working.")
    print("="*60)


if __name__ == "__main__":
    main()
        
        

        