import torch
import pytorch_lightning as pl
from model import UNet
from dataset import UNetDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    logger = TensorBoardLogger('tb_logs', name='unet_model_v0')
    model = UNet(
        n_channels=config.NUM_CHANNELS, 
        n_classes=config.NUM_CLASSES, 
        learning_rate=config.LEARNING_RATE, 
        bilinear=True,
    )
    dm = UNetDataModule(
        imgs_dir=config.IMGS_DIR, 
        masks_dir=config.MASKS_DIR, 
        val_percent=config.VAL_PERCENT, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        profiler='simple',
        logger=logger,
        accelerator=config.ACCELERATOR, 
        devices=config.DEVICES,
        min_epochs=config.MIN_EPOCHS, 
        max_epochs=config.MAX_EPOCHS, 
        precision=config.PRECISION,
        callbacks=[MyPrintingCallback(), EarlyStopping(monitor='validation_loss')],
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)