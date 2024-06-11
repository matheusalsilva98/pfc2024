import numpy as np
import os
from pathlib import Path
from dataset import CBERS4A_CloudDataset
from model import UNet
import config
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch

my_ds = CBERS4A_CloudDataset(imgs_dir=config.IMGS_DIR, masks_dir=config.MASKS_DIR)

val_percent=config.VAL_PERCENT
test_percent=config.TEST_PERCENT

n_val = int(len(my_ds) * val_percent)
n_test = int(len(my_ds) * test_percent)
n_train = len(my_ds) - n_val - n_test
train_ds, val_ds, test_ds = random_split(my_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))

test_dataloader = DataLoader(
                            test_ds,
                            batch_size=config.BATCH_SIZE,
                            num_workers=config.BATCH_SIZE,
                            shuffle=False,
                            )

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
            if image.shape == (4, 512, 512):
                image.transpose([1,2,0])
                image = image[:,:,:3]
            plt.imshow(image)
        fig.subplots_adjust(top=0.8)
        return axarr, fig

def save_inference_to_disk(plot, image_name):
    image_name = Path(image_name).name.split(".")[0]
    output_path = "C:/Users/SE10/Desktop/PFC_reginaldo_dados/pfc2024-main/inferences"
    report_path = os.path.join(
        output_path,
        "report_image_{name}.jpg".format(
            name=image_name,
        ),
    )
    plot.savefig(report_path, format="jpg", bbox_inches="tight")
    return 

def test_inference(dataloader=test_dataloader):

    test_dl = dataloader.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = UNet.load_from_checkpoint(checkpoint_path="C:/Users/SE10/Desktop/PFC_reginaldo_dados/pfc2024-main/model_ckpts/epoch=25-step=1846.ckpt")
    model = model.float()
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            if idx % 10 == 0:
                images, masks = batch['image'], batch['mask']

                images = images.unsqueeze(0)
                images = images.to(device)

                # mask plot preparation
                predicted_mask = model(images)
                predicted_mask = predicted_mask.to("cpu")

                masks = masks.numpy().astype(np.uint8)
                masks = np.squeeze(masks)

                # image plot preparation
                images = images.to("cpu")
                images = images.numpy()
                images = np.squeeze(images)
                images = (images / images.max()) * 255.
                images = images.astype(np.uint8)
                images = images.transpose((1,2,0))
                images = images[:,:,:3]

                # predicted mask plot preparation
                predicted_mask = predicted_mask.numpy()
                predicted_mask = np.squeeze(predicted_mask)
                predicted_mask = np.argmax(predicted_mask, axis=0)
                predicted_mask = predicted_mask.astype(np.uint8)
                
                plot_title = f'batch_{idx}'
                plt_result, fig = generate_visualization(
                    image=images,
                    ground_truth_mask=masks,
                    predicted_mask=predicted_mask)

                save_inference_to_disk(fig, plot_title)

            plt.close(fig)
        return

test_inference()
