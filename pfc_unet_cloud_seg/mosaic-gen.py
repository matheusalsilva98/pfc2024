import numpy as np
from osgeo import gdal
import imageio.v2 as imageio

def CreateGeoTiff(outRaster, data, geo_transform, projection, dtype):
    driver = gdal.GetDriverByName('GTiff')
    no_bands, rows, cols = data.shape
    DataSet = driver.Create(outRaster, cols, rows, no_bands, dtype)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    for i, image in enumerate(data, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    DataSet = None

img1 = imageio.imread('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2021-11-02/11_02_2m.tif')
img2 = imageio.imread('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2021-12-03/12_03_2m.tif')
img3 = imageio.imread('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2022-03-06/22_03_06_2m.tif')
mask1 = imageio.imread('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2021-11-02/treino1/mascara-02-11-2021.tif')
mask2 = imageio.imread('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2021-12-03/treino1/mascara-03-12-2021.tif')
mask3 = imageio.imread('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2022-03-06/treino1/mosaico-2022-03-06.tif')

mosaico = np.zeros((img1.shape[0], img1.shape[1], 3))

for x in range(img1.shape[0]):
    for y in range(img1.shape[1]):
        if mask1[x, y] == 0:
            mosaico[x, y, :3] = img1[x, y, :3]    
        elif mask2[x, y] == 0:
            mosaico[x, y, :3] = img2[x, y, :3]
        elif mask3[x, y] == 0:
            mosaico[x, y, :3] = img3[x, y, :3]
        else:
            mosaico[x, y, :3] = img1[x, y, :3]

mosaico = mosaico.transpose((2, 0, 1))

ds=gdal.Open('/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/2021-11-02/11_02_2m.tif')
CreateGeoTiff(outRaster='/media/reginaldo/pfc-dados/dataset-piloto/cenas-teste/mosaico/mosaico-21-11-02-e-21-12-03-e-22-03-06.tif', data=mosaico,
            geo_transform=ds.GetGeoTransform(), projection=ds.GetProjection(), dtype=gdal.GDT_Float32)
            











