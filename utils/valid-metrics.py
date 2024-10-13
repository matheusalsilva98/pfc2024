import torch
from unet.model import UNet
from unet.dataset import CBERS4A_CloudDataset
from torch.utils.data import DataLoader
import unet.config as config
import imageio.v2 as imageio
import numpy as np
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path
from osgeo import gdal
import seaborn as sn
from PIL import Image
import pandas as pd
import io
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score, MulticlassRecall, ConfusionMatrix
import argparse
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text
    
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

def save_plot_to_disk(plot, image_name, idx, output_path):
    image_name = Path(image_name).name.split(".")[0]
    report_path = os.path.join(
        output_path,
        "report_image_{name}.jpg".format(
            name=image_name
        ),
    )
    plot.savefig(report_path, format="jpg", bbox_inches="tight")
    return report_path

with torch.no_grad():
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = UNet.load_from_checkpoint("/media/reginaldo/pfc-dados/dataset-piloto/modelos-treinados/treino5/epoch=67-step=5848.ckpt").to('cuda')

    list_outputs = []
    list_labels = []

    background_jaccard_index = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average=None)[0].to(device)
    background_accuracy = MulticlassAccuracy(num_classes=config.NUM_CLASSES, average=None)[0].to(device)
    background_precision = MulticlassPrecision(num_classes=config.NUM_CLASSES, average=None)[0].to(device)
    background_f1score = MulticlassF1Score(num_classes=config.NUM_CLASSES, average=None)[0].to(device)
    background_recall = MulticlassRecall(num_classes=config.NUM_CLASSES, average=None)[0].to(device)

    nuvem_densa_jaccard_index = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average=None)[1].to(device)
    nuvem_densa_accuracy = MulticlassAccuracy(num_classes=config.NUM_CLASSES, average=None)[1].to(device)
    nuvem_densa_precision = MulticlassPrecision(num_classes=config.NUM_CLASSES, average=None)[1].to(device)
    nuvem_densa_f1score = MulticlassF1Score(num_classes=config.NUM_CLASSES, average=None)[1].to(device)
    nuvem_densa_recall = MulticlassRecall(num_classes=config.NUM_CLASSES, average=None)[1].to(device)

    nuvem_fina_jaccard_index = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average=None)[2].to(device)
    nuvem_fina_accuracy = MulticlassAccuracy(num_classes=config.NUM_CLASSES, average=None)[2].to(device)
    nuvem_fina_precision = MulticlassPrecision(num_classes=config.NUM_CLASSES, average=None)[2].to(device)
    nuvem_fina_f1score = MulticlassF1Score(num_classes=config.NUM_CLASSES, average=None)[2].to(device)
    nuvem_fina_recall = MulticlassRecall(num_classes=config.NUM_CLASSES, average=None)[2].to(device)

    sombra_jaccard_index = MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average=None)[3].to(device)
    sombra_accuracy = MulticlassAccuracy(num_classes=config.NUM_CLASSES, average=None)[3].to(device)
    sombra_precision = MulticlassPrecision(num_classes=config.NUM_CLASSES, average=None)[3].to(device)
    sombra_f1score = MulticlassF1Score(num_classes=config.NUM_CLASSES, average=None)[3].to(device)
    sombra_recall = MulticlassRecall(num_classes=config.NUM_CLASSES, average=None)[3].to(device)

    parser.add_argument('--local', type=str)
    parser.add_argument('--config', type=str)

    args = parser.parse_args()

    inicio_for = time.time()

    root_diretorio = f'/media/reginaldo/pfc-dados/dataset-principal/dados-valid/{args.local}/{args.configuracao}'
    output_path = f'/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-valid/{args.local}/'

    model = model.float()
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

    for idx, batch in enumerate(valid_dl):
        images, img_paths, masks = batch['image'], batch['image_path'], batch['mask']
        
        for image, img_path, mask in zip(images, img_paths, masks):
            image = image.unsqueeze(0)
            #image = image.to(device)
            image = image.to('cuda')
            predicted_mask = model(image)
            #predicted_mask = np.squeeze(predicted_mask)
            predicted_mask = predicted_mask.squeeze(0)
            predicted_mask = torch.argmax(predicted_mask, dim=0)

            list_outputs.append(predicted_mask.to(device))
            list_labels.append(mask.to(device))

    fim_for = time.time()
    duracao = fim_for - inicio_for
    print(f"Tempo do FOR: {duracao:.5f} seg")


    inicio_metricas = time.time()

    outputs = torch.cat(list_outputs)
    labels = torch.cat(list_labels)

    mean_background_jaccard_index = background_jaccard_index(outputs, labels)
    mean_background_accuracy = background_accuracy(outputs, labels)
    mean_background_precision = background_precision(outputs, labels)
    mean_background_f1score = background_f1score(outputs, labels)
    mean_background_recall = background_recall(outputs, labels)
    mean_nuvem_densa_jaccard_index = nuvem_densa_jaccard_index(outputs, labels)
    mean_nuvem_densa_accuracy = nuvem_densa_accuracy(outputs, labels)
    mean_nuvem_densa_precision = nuvem_densa_precision(outputs, labels)
    mean_nuvem_densa_f1score = nuvem_densa_f1score(outputs, labels)
    mean_nuvem_densa_recall = nuvem_densa_recall(outputs, labels)
    mean_nuvem_fina_jaccard_index = nuvem_fina_jaccard_index(outputs, labels)
    mean_nuvem_fina_accuracy = nuvem_fina_accuracy(outputs, labels)
    mean_nuvem_fina_precision = nuvem_fina_precision(outputs, labels)
    mean_nuvem_fina_f1score = nuvem_fina_f1score(outputs, labels)
    mean_nuvem_fina_recall = nuvem_fina_recall(outputs, labels)
    mean_sombra_jaccard_index = sombra_jaccard_index(outputs, labels)
    mean_sombra_accuracy = sombra_accuracy(outputs, labels)
    mean_sombra_precision = sombra_precision(outputs, labels)
    mean_sombra_f1score = sombra_f1score(outputs, labels)
    mean_sombra_recall = sombra_recall(outputs, labels)

    confusion = ConfusionMatrix(task='multiclass', num_classes=config.NUM_CLASSES).to(device)
    confusion(outputs, labels)
    computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

    _label_ind_by_names = {'background': 0, 'nuvem_densa': 1, 'nuvem_fina': 2, 'sombra': 3}

    # confusion matrix
    df_cm = pd.DataFrame(
        computed_confusion,
        index=_label_ind_by_names.values(),
        columns=_label_ind_by_names.values(),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set_theme(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)

    ax.set_title(f"Matriz de Confusão ({args.local}-{args.configuracao})", fontsize=10)
    ax.legend(
        _label_ind_by_names.values(),
        _label_ind_by_names.keys(),
        handler_map={int: IntHandler()},
        loc='upper left',
        bbox_to_anchor=(1.2, 1)
    )

    fig.savefig(f"/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-valid/{args.local}/{args.configuracao}/matriz_conf_{args.local}_{args.configuracao}.jpg")

    metricas = f'Resultados para {root_diretorio}:\n\nmean_background_jaccard_index: {mean_background_jaccard_index}\nmean_background_accuracy: {mean_background_accuracy}\nmean_background_precision: {mean_background_precision}\nmean_background_f1score: {mean_background_f1score}\nmean_background_recall: {mean_background_recall}\n\nmean_nuvem_densa_jaccard_index: {mean_nuvem_densa_jaccard_index}\nmean_nuvem_densa_accuracy: {mean_nuvem_densa_accuracy}\nmean_nuvem_densa_precision: {mean_nuvem_densa_precision}\nmean_nuvem_densa_f1score: {mean_nuvem_densa_f1score}\nmean_nuvem_densa_recall: {mean_nuvem_densa_recall}\n\nmean_nuvem_fina_jaccard_index: {mean_nuvem_fina_jaccard_index}\nmean_nuvem_fina_accuracy: {mean_nuvem_fina_accuracy}\nmean_nuvem_fina_precision: {mean_nuvem_fina_precision}\nmean_nuvem_fina_f1score: {mean_nuvem_fina_f1score}\nmean_nuvem_fina_recall: {mean_nuvem_fina_recall}\n\nmean_sombra_jaccard_index: {mean_sombra_jaccard_index}\nmean_sombra_accuracy: {mean_sombra_accuracy}\nmean_sombra_precision: {mean_sombra_precision}\nmean_sombra_f1score: {mean_sombra_f1score}\nmean_sombra_recall: {mean_sombra_recall}'

    with open(f"/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-valid/{args.local}/{args.configuracao}/resultados-valid-{args.local}-{args.configuracao}-treino1-teste.txt", "w") as arquivo:
        print(metricas, file=arquivo)

    final_metricas = time.time()
    duracao2 = final_metricas - inicio_metricas
    print(f"Tempo para salvar os resultados: {duracao2:.5f} seg")

    list_labels.clear()
    list_outputs.clear()