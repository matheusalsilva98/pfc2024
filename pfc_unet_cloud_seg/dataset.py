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
from pathlib import Path
from collections import defaultdict

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, train_imgs_dir, train_masks_dir, valid_imgs_dir, valid_masks_dir, batch_size, num_workers):
        super().__init__()
        self.train_imgs_dir = train_imgs_dir
        self.train_masks_dir = train_masks_dir
        self.valid_imgs_dir = valid_imgs_dir
        self.valid_masks_dir = valid_masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.train_ds = CBERS4A_CloudDataset(imgs_dir=self.train_imgs_dir, masks_dir=self.train_masks_dir)
        self.valid_ds = CBERS4A_CloudDataset(imgs_dir=self.valid_imgs_dir, masks_dir=self.valid_masks_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=config.PREFETCH_FACTOR,
            shuffle=True,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=config.PREFETCH_FACTOR,
            shuffle=False,
            persistent_workers=True,
            drop_last=True,
        )

class CBERS4A_CloudDataset(Dataset):
  def __init__(self, imgs_dir, masks_dir):
    self.imgs_dir = imgs_dir
    self.masks_dir = masks_dir

    self.ids = [splitext(file)[0].split('_')[-1] for file in listdir(imgs_dir)
                if not file.startswith('.')]
    
    self.img_dict = self.build_image_dict(self.imgs_dir)
    self.mask_dict = self.build_image_dict(self.masks_dir)
  
  def build_image_dict(self, root_dir):
    output_dict = defaultdict(list)
    for p in Path(root_dir).rglob("*.tif"):
        key = str(p.stem).split("_")[-1]
        output_dict[key].append(str(p))
    return output_dict

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, i):
    idx = self.ids[i]
    img_file = self.img_dict.get(idx, [])
    mask_file = self.mask_dict.get(idx, [])

    assert len(mask_file) == 1, \
        f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'

    img = imageio.imread(img_file[0])
    mask = imageio.imread(mask_file[0])

    assert img.shape == (512, 512, 6), f"Image with id {idx} with dims {img.shape}. Path to image: {img_file}"
    assert mask.shape == (512, 512), f"Mask with id {idx} with dims {mask.shape}. Path to image: {mask_file}"

    img = img.transpose((2, 0, 1))

    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)

    img[:4,:,:] = nn.functional.normalize(img[:4,:,:])
    return {
        'image': img,
        'mask': mask
    }
