from pytorch_lightning.callbacks import EarlyStopping, Callback
import numpy as np
import os
from pathlib import Path
import imageio
import config
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import torch

class MyPrintingCallback(Callback):
    def __init__(self, save_outputs=True, output_path=None):
        super().__init__()
        self.save_outputs = save_outputs
        self.output_path = output_path

    @staticmethod
    def generate_visualization(fig_title=None, fig_size=None, font_size=16, **images):
        n = len(images)
        fig_size = (16, 5) if fig_size is None else fig_size
        fig, axarr = plt.subplots(1, n, figsize=fig_size)
        if fig_title is not None:
            fig.suptitle(fig_title, fontsize=font_size)
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(" ".join(name.split("_")).title())
            if image.shape == (config.NUM_CHANNELS, config.PATCH_SIZE, config.PATCH_SIZE):
                image.transpose([1,2,0])
                image = image[:,:,:3]
            plt.imshow(image)
        fig.subplots_adjust(top=0.8)
        return axarr, fig
    
    def log_data_to_tensorboard(self, saved_image, image_path, logger, current_epoch):
        data = imageio.imread(saved_image)
        data = np.moveaxis(data, -1, 0)
        data = torch.from_numpy(data)
        logger.experiment.add_image(image_path, data, current_epoch)

    def save_plot_to_disk(self, plot, image_name, current_epoch):
        image_name = str(Path(image_name).name.split(".")[0])
        report_path = os.path.join(
            self.output_path,
            "report_image_{name}_epoch_{epoch}.jpg".format(
                name=image_name,
                epoch=current_epoch,
            ),
        )
        plot.savefig(report_path, format="jpg", bbox_inches="tight")
        return report_path

    def on_sanity_check_end(self, trainer, pl_module):
        self.save_outputs = True
        self.output_path = os.path.join(trainer.log_dir, "image_logs")
        if not os.path.exists(self.output_path):
            Path(self.output_path).mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if not self.save_outputs:
            return
        val_dl = trainer.datamodule.val_dataloader().dataset
        device = pl_module.device
        logger = trainer.logger
        
        for idx, batch in enumerate(val_dl):
            if idx % 10 == 0:
                image, mask = batch['image'], batch['mask']

                image = image.unsqueeze(0)
                image = image.to(device)

                # mask plot preparation
                predicted_mask = pl_module(image)
                predicted_mask = predicted_mask.to("cpu")

                mask = mask.numpy().astype(np.uint8)
                mask = np.squeeze(mask)

                # image plot preparation
                image = image.to("cpu")
                image = image.numpy()
                image = np.squeeze(image)
                image = image.transpose((1,2,0))
                image = image[:,:,:3]
                image = (image / image.max()) * 255.
                image = image.astype(np.uint8)

                # predicted mask plot preparation
                predicted_mask = predicted_mask.numpy()
                predicted_mask = np.squeeze(predicted_mask)
                predicted_mask = np.argmax(predicted_mask, axis=0)
                predicted_mask = predicted_mask.astype(np.uint8)
                
                plot_title = f'batch_{idx}'
                plt_result, fig = self.generate_visualization(
                    fig_title=plot_title,
                    fig_size=None,
                    font_size=16,
                    image=image,
                    ground_truth_mask=mask,
                    predicted_mask=predicted_mask,
                )
                if self.save_outputs:
                    saved_image = self.save_plot_to_disk(
                        fig, plot_title, trainer.current_epoch
                    )
                    self.log_data_to_tensorboard(
                        saved_image, plot_title, logger, trainer.current_epoch
                    )
                plt.close(fig)
        return
