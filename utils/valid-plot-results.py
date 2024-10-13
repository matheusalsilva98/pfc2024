import torch
from unet.model import UNet
from unet.dataset import CBERS4A_CloudDataset
from torch.utils.data import DataLoader
import unet.config as config
import imageio.v2 as imageio
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path
from osgeo import gdal
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score, MulticlassRecall
import argparse

parser = argparse.ArgumentParser()

def generate_visualization(image, mask, pred, fig_title=None, fig_size=None, font_size=16):
    fig_size = (15, 5) if fig_size is None else fig_size
    fig, axarr = plt.subplots(1, 3, figsize=fig_size)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=font_size)
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Recorte de imagem da cena')
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Máscara de referência vetorizada manualmente')
    # Get unique values in the segmentation map
    vals = np.unique(mask)

    # Use a predefined colormap with enough colors (tab10 in this case)
    cmap = plt.cm.get_cmap(name=None, lut=4)  # colormap for 4 classes
    colors = [cmap(i) for i in range(4)]  # Always have 4 colors for 4 possible classes
    labels = ['Background', 'Nuvem Densa', 'Nuvem Fina', 'Sombra']

    # Only keep colors and labels for the classes that exist in the mask
    colors = [colors[i] for i in vals]
    labels = [labels[i] for i in vals]

    assert len(labels) == len(colors)

    # Create the colormap for displaying the image
    cmap = mpl.colors.ListedColormap(colors)
    plt.imshow(mask, cmap=cmap)

    # Create patches for the legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(vals))]
    plt.legend(handles=patches)

    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Inferência gerada pelos pesos do Treino 1')
    # Get unique values in the segmentation map
    vals = np.unique(pred)

    # Use a predefined colormap with enough colors (tab10 in this case)
    cmap = plt.cm.get_cmap(name=None, lut=4)  # colormap for 4 classes
    colors = [cmap(i) for i in range(4)]  # Always have 4 colors for 4 possible classes
    labels = ['Background', 'Nuvem Densa', 'Nuvem Fina', 'Sombra']

    # Only keep colors and labels for the classes that exist in the mask
    colors = [colors[i] for i in vals]
    labels = [labels[i] for i in vals]

    assert len(labels) == len(colors)

    # Create the colormap for displaying the image
    cmap = mpl.colors.ListedColormap(colors)
    plt.imshow(pred, cmap=cmap)

    # Create patches for the legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(vals))]
    plt.legend(handles=patches)

    fig.subplots_adjust(top=0.8)
    return axarr, fig

def save_plot_to_disk(plot, image_name, idx, output_path):
    image_name = Path(image_name).name.split(".")[0]
    report_path = os.path.join(
        output_path,
        "{name}.jpg".format(
            name=image_name
        ),
    )
    plot.savefig(report_path, format="jpg", bbox_inches="tight")
    return report_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet.load_from_checkpoint("/media/reginaldo/pfc-dados/dataset-piloto/modelos-treinados/treino5/epoch=67-step=5848.ckpt")

parser.add_argument('--local', type=str)

args = parser.parse_args()

root_diretorio = f'/media/reginaldo/pfc-dados/dataset-principal/dados-valid/{args.local}'
output_path = f'/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-valid/{args.local}/'

model = model.float()
model = model.to(device)
model.eval()
valid_ds = CBERS4A_CloudDataset(root_dir=root_diretorio, train=False)
valid_dl = DataLoader(
            valid_ds,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            prefetch_factor=config.PREFETCH_FACTOR,
            shuffle=False,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            drop_last=True,
        )

saved_paths = set()

with torch.no_grad():
    for idx, batch in enumerate(valid_dl):
        if len(saved_paths) == 8:
            break

        images, img_paths, masks = batch['image'], batch['image_path'], batch['mask']
        
        for image, img_path, mask in zip(images, img_paths, masks):
            folder_path = os.path.dirname(img_path)

            if folder_path not in saved_paths:
                image = image.unsqueeze(0)
                image = image.to(device)
                predicted_mask = model(image)
                predicted_mask = predicted_mask.to("cpu")

                # mask plot preparation
                mask = mask.numpy().astype(np.uint8)
                mask = np.squeeze(mask)

                image_vis = imageio.imread(img_path)[:,:,:3]
                image_vis = (image_vis / image_vis.max()) * 255.
                image_vis = image_vis.astype(np.uint8)

                # predicted mask plot preparation
                predicted_mask = predicted_mask.numpy()
                predicted_mask = np.squeeze(predicted_mask)
                predicted_mask = np.argmax(predicted_mask, axis=0)
                predicted_mask = predicted_mask.astype(np.uint8)

                plot_title = f''
                plt_result, fig = generate_visualization(
                    fig_title=plot_title,
                    fig_size=None,
                    font_size=16,
                    image=image_vis,
                    mask=mask,
                    pred=predicted_mask
                )
                
                save_plot_to_disk(
                    fig, 
                    plot_title, 
                    idx,
                    output_path
                )
                plt.close(fig)
        
                saved_paths.add(folder_path)


saved_paths.clear()