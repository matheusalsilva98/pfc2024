import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from glob import glob
from os.path import splitext
from os import listdir
import imageio.v2 as imageio
import numpy as np
from torch.utils.data import DataLoader, random_split
import config

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, imgs_dir, masks_dir, val_percent, batch_size, num_workers):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        my_ds = CBERS4A_CloudDataset(imgs_dir=self.imgs_dir, masks_dir=self.masks_dir)
        
        n_val = int(len(my_ds) * self.val_percent)
        n_train = len(my_ds) - n_val
        self.train_ds, self.val_ds = random_split(my_ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

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

class CBERS4A_CloudDataset(Dataset):
  def __init__(self, imgs_dir, masks_dir, scale=1, mask_preffix='mask_'):
    self.imgs_dir = imgs_dir
    self.masks_dir = masks_dir
    self.scale = scale
    self.mask_preffix = mask_preffix
    assert 0 < scale <= 1, 'Scale must be between 0 and 1'

    self.ids = [splitext(file)[0].split('_')[1] for file in listdir(imgs_dir)
                if not file.startswith('.')]

  def __len__(self):
    return len(self.ids)

  @classmethod
  def preprocess(cls, imageio_img, scale):
    w = imageio_img.shape[0]
    h = imageio_img.shape[1]
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small'
    if len(imageio_img.shape) > 2:
      imageio_img = imageio_img.transpose((2, 0, 1))
      if imageio_img.max() > 1:
        imageio_img = imageio_img / imageio_img.max()
    return imageio_img

  def __getitem__(self, i):
    idx = self.ids[i]
    mask_file = glob(self.masks_dir + '/' + self.mask_preffix + idx + '*.tif')
    img_file = glob(self.imgs_dir + '/' +'image_'+ idx + '*.tif')

    assert len(mask_file) == 1, \
        f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
    assert len(img_file) == 1, \
        f'Either no image or multiple images found for the ID {idx}: {img_file}'

    img = imageio.imread(img_file[0])
    mask = imageio.imread(mask_file[0])

    img = self.preprocess(img, self.scale)
    mask = self.preprocess(mask, self.scale)

    img = np.float32(img)
    mask = mask.astype('uint8')

    return {
        'image': torch.from_numpy(img),
        'mask': torch.from_numpy(mask)
    }