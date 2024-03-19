import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
from os.path import splitext
from os import listdir
import imageio.v2 as imageio
from torch.utils.data import DataLoader, random_split
import config
import numpy as np

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, imgs_dir, masks_dir, val_percent, test_percent, batch_size, num_workers):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        my_ds = CBERS4A_CloudDataset(imgs_dir=self.imgs_dir, masks_dir=self.masks_dir)

        n_val = int(len(my_ds) * self.val_percent)
        n_test = int(len(my_ds) * self.test_percent)
        n_train = len(my_ds) - n_val - n_test
        self.train_ds, self.val_ds, self.test_ds = random_split(my_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=config.PREFETCH_FACTOR,
            shuffle=True,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=config.PREFETCH_FACTOR,
            shuffle=False,
            persistent_workers=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

class CBERS4A_CloudDataset(Dataset):
  def __init__(self, imgs_dir, masks_dir):
    self.imgs_dir = imgs_dir
    self.masks_dir = masks_dir

    self.ids = [splitext(file)[0].split('_')[1] for file in listdir(imgs_dir)
                if not file.startswith('.')]

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, i):
    idx = self.ids[i]
    mask_file = glob(self.masks_dir + '/' + 'mask_' + idx + '*.tif')
    img_file = glob(self.imgs_dir + '/' +'image_'+ idx + '*.tif')

    assert len(mask_file) == 1, \
        f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'

    img = imageio.imread(img_file[0])
    mask = imageio.imread(mask_file[0])

    img = img.transpose((2, 0, 1))

    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)

    img = nn.functional.normalize(img)
    return {
        'image': img,
        'mask': mask
    }
