import torch
import pytorch_lightning as pl
from model import UNet
from dataset import UNetDataModule
import config_servidor as config
from callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_segmentation_models_trainer.custom_callbacks.image_callbacks import ImageSegmentationResultCallback

# torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    logger = TensorBoardLogger('tb_logs', name='unet_model_v0')
    model = UNet(
        n_channels=config.NUM_CHANNELS, 
        n_classes=config.NUM_CLASSES, 
        learning_rate=config.LEARNING_RATE, 
        bilinear=True,
    )
    model = model.float()
    dm = UNetDataModule(
        imgs_dir=config.IMGS_DIR, 
        masks_dir=config.MASKS_DIR, 
        val_percent=config.VAL_PERCENT,
        test_percent=config.TEST_PERCENT, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(
        # profiler='simple',
        logger=logger,
        strategy="ddp",
        gpus=3,
        precision=32,
        min_epochs=config.MIN_EPOCHS, 
        max_epochs=config.MAX_EPOCHS, 
        # precision=config.PRECISION,
        callbacks=[
            MyPrintingCallback(),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(
                monitor='validation_jaccard_index',
                patience=10,
                mode="max"
            ),
            StochasticWeightAveraging(swa_lrs=1e-3),
        ],
        sync_batchnorm=True,
    )
    trainer.fit(model, dm)
    trainer.validate(model, dm)
