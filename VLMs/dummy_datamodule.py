from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dummy_dataset import DummyDataset

class DummyDataModule(LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = DummyDataset()
        self.val_dataset = DummyDataset(length=20)
        self.test_dataset = DummyDataset(length=20)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
