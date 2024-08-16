import torch
import pytorch_lightning as pl
from model import UNet
from dataset import UNetDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

# torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    logger = TensorBoardLogger('tb_logs', name='unet_model_v1_6bandas')
    model = UNet(
        n_channels=config.NUM_CHANNELS, 
        n_classes=config.NUM_CLASSES, 
        learning_rate=config.LEARNING_RATE, 
        bilinear=True,
    )
    model = model.float()
    dm = UNetDataModule(
        train_imgs_dir=config.TRAIN_IMGS_DIR,
        train_masks_dir=config.TRAIN_MASKS_DIR,
        valid_imgs_dir=config.VALID_IMGS_DIR,
        valid_masks_dir=config.VALID_MASKS_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        use_augmentations=config.USE_AUGMENTATIONS,
    )
    trainer = pl.Trainer(
        # profiler='simple',
        logger=logger,
        strategy="ddp",
        accelerator=config.ACCELERATOR, 
        devices=config.DEVICES,
        precision=config.PRECISION,
        min_epochs=config.MIN_EPOCHS, 
        max_epochs=config.MAX_EPOCHS, 
        # precision=config.PRECISION,
        callbacks=[
            MyPrintingCallback(output_path=config.LOG_OUTPUT_PATH),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(
                monitor='validation_jaccard_index',
                patience=10,
                mode="max"
            ),
            StochasticWeightAveraging(swa_lrs=1e-3, swa_epoch_start=0.2),
        ],
        sync_batchnorm=True,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
