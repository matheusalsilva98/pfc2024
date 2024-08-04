import imageio.v2 as imageio
from osgeo import gdal
import numpy as np
import os

def CreateGeoTiff(outRaster, data, geo_transform, projection, dtype):
    driver = gdal.GetDriverByName('GTiff')
    no_bands, rows, cols = data.shape
    DataSet = driver.Create(outRaster, cols, rows, no_bands, dtype)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    for i, image in enumerate(data, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    DataSet = None

imgs_dir = '/media/reginaldo/pfc-dados/dataset-piloto/dados-validacao/images_LOCAL_CONFIG'

for img in os.listdir(imgs_dir):
    img_dir = imgs_dir + '/' + img
    image = imageio.imread(img_dir)

    red_img_transp = image[:,:,0]
    green_img_transp = image[:,:,1]
    blue_img_transp = image[:,:,2]
    nir_img_transp = image[:,:,3]
    # NDVI band
    ndvi_img = (nir_img_transp - red_img_transp) / (nir_img_transp + red_img_transp + 1e-6)
    ndvi_img = ndvi_img.astype('float32')
    # WI band
    M = (blue_img_transp + green_img_transp + red_img_transp) / 3 + 1e-6
    wi_img = (abs(red_img_transp - M) + abs(green_img_transp - M) + abs(blue_img_transp - M))/M
    wi_img = wi_img.astype('float32')

    img_array_final = np.dstack((image,ndvi_img))
    img_array_final = np.dstack((img_array_final,wi_img))

    img_array_final = img_array_final.transpose((2, 0, 1))

    ds=gdal.Open(img_dir)

    CreateGeoTiff(outRaster='/media/reginaldo/pfc-dados/dataset-piloto/dados-validacao/images_LOCAL_CONFIG_6bands/'+img_dir.split('/')[-1], data=img_array_final,
                geo_transform=ds.GetGeoTransform(), projection=ds.GetProjection(), dtype=gdal.GDT_Float32)