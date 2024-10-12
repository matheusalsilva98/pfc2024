import torch
import pytorch_lightning as pl
from unet.model import UNet
from unet.dataset import UNetDataModule
import unet.config as config
from unet.callbacks import MyPrintingCallback, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger

# torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    logger = TensorBoardLogger('tb_logs', name='unet_model_v1_6bandas')
    if hasattr(config, "CHECKPOINT"):
        print(f"Resuming from {config.CHECKPOINT}")
        model = UNet.load_from_checkpoint(
            config.CHECKPOINT,
            root_dir=config.DATASET_ROOT_DIR,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            use_augmentations=config.USE_AUGMENTATIONS,
        )
    else:
        model = UNet(
            n_channels=config.NUM_CHANNELS, 
            n_classes=config.NUM_CLASSES, 
            learning_rate=config.LEARNING_RATE, 
            bilinear=True,
        )
    model = model.float()
    dm = UNetDataModule(
        root_dir=config.DATASET_ROOT_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        use_augmentations=config.USE_AUGMENTATIONS,
    )
    dm.setup(None)
    steps_per_epoch = len(dm.train_dataloader())
    model.set_steps_per_epoch(steps_per_epoch)

    trainer = pl.Trainer(
        # profiler='simple',
        logger=logger,
        strategy="ddp",
        accelerator=config.ACCELERATOR, 
        devices=config.DEVICES,
        precision=config.PRECISION,
        min_epochs=config.MIN_EPOCHS, 
        max_epochs=config.MAX_EPOCHS, 
        callbacks=[
            MyPrintingCallback(output_path=config.LOG_OUTPUT_PATH),
            LearningRateMonitor(logging_interval='step'),
            EarlyStopping(
                monitor='val_background_jaccard_index',
                patience=10,
                mode="max"
            ),
            StochasticWeightAveraging(swa_lrs=1e-3),
        ],
        sync_batchnorm=True,
    )
    trainer.fit(model, dm)
    # trainer.validate(model, dm)
