from typing import Any, Dict, Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from os.path import join
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from src.data.bids_loader import BIDSLoader


class AtlasDataset(Dataset):

    def __init__(self, data_dir, is_train=True):
        self.is_train = is_train
        self.prepare_data(data_dir, is_train)

    def prepare_data(self, data_dir, is_train):
        """
        Get the BIDS dataloader
        """
        training = {
            "batch_size": 5,
            "dir_name": join(data_dir, "train"),
            "data_entities": [{"subject": "", "session": "", "suffix": "T1w"}],
            "target_entities": [{"label": "L", "desc": "T1lesion", "suffix": "mask"}],
            "data_derivatives_names": ["ATLAS"],
            "target_derivatives_names": ["ATLAS"],
            "label_names": ["not lesion", "lesion"],
        }
        self.bids_loader = BIDSLoader(
            root_dir=training["dir_name"],
            data_entities=training["data_entities"],
            target_entities=training["target_entities"],
            data_derivatives_names=training["data_derivatives_names"],
            target_derivatives_names=training["target_derivatives_names"],
            label_names=training["label_names"],
            batch_size=training["batch_size"],
        )

    def __len__(self):
        return len(self.bids_loader)

    def __getitem__(self, index):
        images, targets = self.bids_loader.load_sample(index)
        return (images, targets)


class DataModule(LightningDataModule):
    """Example preprocessing and batching poses

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train = AtlasDataset(data_dir, is_train=True)
        self.data_val = AtlasDataset(data_dir, is_train=False)

    @property
    def n_features(self):
        return 4

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        # if not self.data_train and not self.data_val:
        #     # Training and Val set
        #     self.data_train, self.data_val = random_split(
        #         dataset=self.data,
        #         lengths=[int(self.train_val_split[0]*len(self.data)), int(self.train_val_split[1]*len(self.data))],
        #         generator=torch.Generator().manual_seed(42),
        #     )

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = DataModule()
    a = _.train_dataloader()
    c = next(iter(a))
    b = _.val_dataloader()
