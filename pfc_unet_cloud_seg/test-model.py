import torch
from model import UNet
import imageio.v2 as imageio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path
from osgeo import gdal
from torchmetrics.classification import MulticlassJaccardIndex

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UNet.load_from_checkpoint("/media/reginaldo/pfc-dados/dataset-piloto/treino5/epoch=67-step=5848.ckpt")

imgs_dir = '/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2022-03-06/recortes-6b'
jaccard_index = 0.
model = model.float()
model.eval()
len = len(os.listdir(imgs_dir))
with torch.no_grad():
    for idx, img in enumerate(os.listdir(imgs_dir)):
        img_dir = imgs_dir + '/' + img
        image = imageio.imread(img_dir)
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        
        # 4 bands image
        image = nn.functional.normalize(image)

        ## 6 bands image
        #image = nn.functional.normalize(image[:4,:,:])

        image = image.unsqueeze(0)
        image = image.to(device)

        predicted_mask = model(image)
        predicted_mask = predicted_mask.to("cpu")
        predicted_mask = predicted_mask.numpy()
        predicted_mask = np.squeeze(predicted_mask)
        predicted_mask = np.argmax(predicted_mask, axis=0)
        predicted_mask = predicted_mask.astype(np.uint8)

        #predicted_mask = torch.from_numpy(predicted_mask)

        #mask_dir = '/media/reginaldo/pfc-dados/dataset-piloto/real-masks/' + img
        #mask = imageio.imread(mask_dir)
        #mask = mask.astype(np.float32)
        #mask = torch.from_numpy(mask)

        # Creation of mask with threshold

        #predicted_mask = model(image)
        #softmax = nn.Softmax()
        #predicted_mask = softmax(predicted_mask)
        #predicted_mask = predicted_mask.to("cpu")
        #predicted_mask = predicted_mask.numpy()
        #predicted_mask = np.squeeze(predicted_mask)
        #mask_threshold = np.zeros(predicted_mask.shape)
        #mask_threshold[predicted_mask >= 0.80] = 1
        #mask_threshold = np.argmax(mask_threshold, axis=0)
        #mask_threshold = mask_threshold.astype(np.uint8)
        #print(np.unique(mask_threshold))

        img_dir2 = '/media/reginaldo/pfc-dados/dataset-piloto/treino5/06-03-22/recortes'

        ds = gdal.Open(img_dir)

        driver = gdal.GetDriverByName('GTiff')
        rows, cols = predicted_mask.shape
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
    
        DataSet = driver.Create(img_dir2+'/'+'model_mask_'+img[6:11], cols, rows, 1, gdal.GDT_Byte)
        DataSet.SetGeoTransform(geo_transform)
        DataSet.SetProjection(projection)
    
        DataSet.GetRasterBand(1).WriteArray(predicted_mask)
        DataSet = None