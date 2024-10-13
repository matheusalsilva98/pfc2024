import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()

def generate_visualization(image, mask1, mask2, fig_title=None, fig_size=None, font_size=64):
    fig_size = (20, 5) if fig_size is None else fig_size
    fig, axarr = plt.subplots(1, 3, figsize=fig_size)
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=font_size)
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Recorte da cena', fontsize=font_size)
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Inferência para o Treino 1', fontsize=font_size)
    # Get unique values in the segmentation map
    vals1 = np.unique(mask1)

    # Use a predefined colormap with enough colors (tab10 in this case)
    cmap = plt.cm.get_cmap(name=None, lut=4)  # colormap for 4 classes
    colors = [cmap(i) for i in range(4)]  # Always have 4 colors for 4 possible classes
    labels = ['Background', 'Nuvem Densa', 'Nuvem Fina', 'Sombra']

    # Only keep colors and labels for the classes that exist in the mask
    colors = [colors[i] for i in vals1]
    labels = [labels[i] for i in vals1]

    assert len(labels) == len(colors)

    # Create the colormap for displaying the image
    cmap = mpl.colors.ListedColormap(colors)
    plt.imshow(mask1, cmap=cmap)

    # Create patches for the legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(vals1))]
    plt.legend(handles=patches)

    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('Inferência para o Treino 4', fontsize=font_size)
    # Get unique values in the segmentation map
    vals2 = np.unique(mask2)

    # Use a predefined colormap with enough colors (tab10 in this case)
    cmap2 = plt.cm.get_cmap(name=None, lut=4)  # colormap for 4 classes
    colors2 = [cmap2(i) for i in range(4)]  # Always have 4 colors for 4 possible classes
    labels = ['Background', 'Nuvem Densa', 'Nuvem Fina', 'Sombra']

    # Only keep colors and labels for the classes that exist in the mask
    colors2 = [colors2[i] for i in vals2]
    labels = [labels[i] for i in vals2]

    assert len(labels) == len(colors2)

    # Create the colormap for displaying the image
    cmap2 = mpl.colors.ListedColormap(colors2)
    plt.imshow(mask2, cmap=cmap2)

    # Create patches for the legend
    patches2 = [mpatches.Patch(color=colors2[i], label=labels[i]) for i in range(len(vals2))]
    plt.legend(handles=patches2)

    fig.subplots_adjust(top=0.8)
    return axarr, fig

def save_plot_to_disk(plot, cena, data, image_name, idx, output_path):
    image_name = Path(image_name).name.split(".")[0]
    report_path = os.path.join(
        output_path,
        "recorte-comparacao-1e4-{cena}-{data}_{name}.jpg".format(
            name=image_name,
            cena=cena,
            data=data
        ),
    )
    plot.savefig(report_path, format="jpg", bbox_inches="tight")
    return report_path

parser.add_argument('--local', type=str)
parser.add_argument('--data', type=str)

args = parser.parse_args()
#cena = 'sp'
#data = '20200427'
imgs_diretorio = f'/media/reginaldo/pfc-dados/dataset-principal/dados-teste/{args.local}/testes/recortes_{args.data}'
model_masks1_diretorio = f'/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-teste/{args.local}/recortes_{args.data}'
model_masks2_diretorio = f'/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino4/resultados-teste/{args.local}/recortes_{args.data}'
output_path = f'/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/comparacoes/treinos1e4/{args.local}/'

for idx, img_name in enumerate(os.listdir(imgs_diretorio)):
    img_dir = imgs_diretorio + '/' + img_name
    image_vis = imageio.imread(img_dir)
    image_vis = image_vis[:,:,:3]
    image_vis = (image_vis / image_vis.max()) * 255.
    image_vis = image_vis.astype(np.uint8)

    # predicted mask plot preparation from Treino 1
    mask1_dir = model_masks1_diretorio + '/model_mask_' + img_name.split('_')[-1]
    mask1_vis = imageio.imread(mask1_dir)

    # predicted mask plot preparation from Treino 2
    mask2_dir = model_masks2_diretorio + '/model_mask_' + img_name.split('_')[-1]
    mask2_vis = imageio.imread(mask2_dir)

    # plot_title = f'Exemplo de resultado em recorte da cena de 27/04/2020 de SP'
    plt_result, fig = generate_visualization(
        image=image_vis,
        mask1=mask1_vis,
        mask2=mask2_vis,
        fig_title=None,
        fig_size=None,
        font_size=16
    )

    save_plot_to_disk(
        fig,
        args.local,
        args.data,
        f'{idx+1}', 
        idx,
        output_path
    )
    plt.close(fig)