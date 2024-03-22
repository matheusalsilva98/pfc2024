from pytorch_lightning.callbacks import EarlyStopping, Callback
import numpy as np
import os
import Path
import datatime
import imageio
import matplotlib.pyplot as plt
import torch

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def prepare_mask_to_plot(self, mask):
        return np.squeeze(mask).astype(np.uint8)

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
            plt.imshow(image)
        fig.subplots_adjust(top=0.8)
        return axarr, fig
    
    def log_data_to_tensorboard(self, saved_image, image_path, logger, current_epoch):
        data = imageio.imread(saved_image)
        data = np.moveaxis(data, -1, 0)
        data = torch.from_numpy(data)
        logger.experiment.add_image(image_path, data, current_epoch)

    def save_plot_to_disk(self, plot, image_name, current_epoch):
        image_name = Path(image_name).name.split(".")[0]
        report_path = os.path.join(
            self.output_path,
            "report_image_{name}_epoch_{epoch}_{date}.png".format(
                name=image_name,
                date=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                epoch=current_epoch,
            ),
        )
        plot.savefig(report_path, format="png", bbox_inches="tight")
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
        val_ds = pl_module.val_dataloader().dataset
        device = pl_module.device
        logger = trainer.logger
        self.n_samples = (
            pl_module.val_dataloader().batch_size
            if self.n_samples is None
            else self.n_samples
        )
        for i in range(self.n_samples):
            image, mask = val_ds[i].values()
            image = image.unsqueeze(0)
            image = image.to(device)
            predicted_mask = pl_module(image)
            predicted_mask = predicted_mask.to("cpu")
            plot_title = val_ds.get_path(i)
            plt_result, fig = generate_visualization(
                fig_title=plot_title,
                ground_truth_mask=self.prepare_mask_to_plot(mask.numpy()),
                predicted_mask=self.prepare_mask_to_plot(predicted_mask.numpy()),
            )
            if self.save_outputs:
                saved_image = self.save_plot_to_disk(
                    plt_result, plot_title, trainer.current_epoch
                )
                self.log_data_to_tensorboard(
                    saved_image, plot_title, logger, trainer.current_epoch
                )
            plt.close(fig)
        return
