from typing import Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
from os.path import splitext
from os import listdir
import imageio.v2 as imageio
from torch.utils.data import DataLoader, random_split
import unet.config as config
import numpy as np
from pathlib import Path
from collections import defaultdict
import albumentations as A

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers, use_augmentations=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentations = use_augmentations

    def setup(self, stage):
        self.train_ds = CBERS4A_CloudDataset(root_dir=self.root_dir, train=True, use_augmentations=self.use_augmentations)
        self.valid_ds = CBERS4A_CloudDataset(root_dir=self.root_dir, train=False)

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
  def __init__(self, root_dir, train=True, use_augmentations=False):
    self.img_dict, self.mask_dict = self.build_image_dict(root_dir, dataset_type_key="treino" if train else "valid")

    self.ids = [i for i in self.img_dict.keys() if i in self.mask_dict and len(self.mask_dict[i]) > 0]
    
    self.transform = A.Compose(
        [
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
    ) if use_augmentations else None
  
  def build_image_dict(self, root_dir: str, dataset_type_key: str) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """_summary_

    Args:
        root_dir (str): Diretório raíz
        dataset_type_key (str): treino ou valid
        as imagens estão nos diretórios imgs e as máscaras nos diretórios masks
    Returns:
        Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]: retorna os dicionários de imagens e de máscaras
    """
    image_output_dict = defaultdict(list)
    mask_output_dict = defaultdict(list)
    for p in Path(root_dir).rglob("*.tif"):
        key = str(p.stem)
        if dataset_type_key not in str(p):
            continue
        if "imgs" in str(p):
            image_output_dict[key].append(str(p))
        else:
            mask_output_dict[key.replace("_mask_", "_image_")].append(str(p))
    return image_output_dict, mask_output_dict

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

    assert img.shape == (config.PATCH_SIZE, config.PATCH_SIZE, config.NUM_CHANNELS), f"Image with id {idx} with dims {img.shape} instead of {(config.PATCH_SIZE, config.PATCH_SIZE, config.NUM_CHANNELS)}. Path to image: {img_file}"
    assert mask.shape == (config.PATCH_SIZE, config.PATCH_SIZE), f"Mask with id {idx} with dims {mask.shape} instead of {(config.PATCH_SIZE, config.PATCH_SIZE)}. Path to image: {mask_file}"

    if self.transform is not None:
        output = self.transform(image=img, mask=mask)
        img, mask = output["image"], output["mask"]

    img = img.transpose((2, 0, 1))

    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)

    img[:4,:,:] = nn.functional.normalize(img[:4,:,:])

    return {
        'image': img,
        'image_path': img_file[0],
        'mask': mask,
        'mask_path': mask_file[0]
    }
