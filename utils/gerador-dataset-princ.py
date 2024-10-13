import numpy as np
import imageio.v2 as imageio
from osgeo import gdal
import os
import torch
from SatelliteCloudGenerator import *

def CreateGeoTiff(outRaster, data, geo_transform, projection, dtype):
    driver = gdal.GetDriverByName('GTiff')
    no_bands, rows, cols = data.shape
    DataSet = driver.Create(outRaster, cols, rows, no_bands, dtype)
    DataSet.SetGeoTransform(geo_transform)
    DataSet.SetProjection(projection)

    for i, image in enumerate(data, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    DataSet = None

def crop_images(img_dir, patch_size, output_dir):

    # Recorte da imagem em patches de 512x512
    cbers4a_image = gdal.Open(img_dir)
    gt = cbers4a_image.GetGeoTransform()

    # get coordinates of upper left corner
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]

    # determine total length of raster
    xlen = res * round((cbers4a_image.RasterXSize//patch_size)*patch_size)
    ylen = res * round((cbers4a_image.RasterYSize//patch_size)*patch_size)

    # number of tiles in x and y direction
    xdiv = round((cbers4a_image.RasterXSize//patch_size)*patch_size / patch_size)
    ydiv = round((cbers4a_image.RasterYSize//patch_size)*patch_size / patch_size)

    # size of a single tile
    xsize = xlen/xdiv
    ysize = ylen/ydiv

    # create lists of x and y coordinates
    xsteps = [xmin + xsize * i for i in range(xdiv+1)]
    ysteps = [ymax - ysize * i for i in range(ydiv+1)]

    # loop over min and max x and y coordinates
    count = 1
    for i in range(xdiv):
        for j in range(ydiv):
            xmin = xsteps[i]
            xmax = xsteps[i+1]
            ymax = ysteps[j]
            ymin = ysteps[j+1]
            
            if count <= 9:
                # gdal translate to subset the input raster
                gdal.Translate(output_dir+f"/image_0000"+str(count)+".tif", cbers4a_image, projWin = (xmin, ymax, xmax, ymin), xRes = res, yRes = -res)
                count += 1
            elif (count >= 10 and count <= 99):
                # gdal translate to subset the input raster
                gdal.Translate(output_dir+f"/image_000"+str(count)+".tif", cbers4a_image, projWin = (xmin, ymax, xmax, ymin), xRes = res, yRes = -res)
                count += 1
            elif (count >= 100 and count <= 999):
                # gdal translate to subset the input raster
                gdal.Translate(output_dir+f"/image_00"+str(count)+".tif", cbers4a_image, projWin = (xmin, ymax, xmax, ymin), xRes = res, yRes = -res)
                count += 1
            elif (count >= 1000 and count <= 9999):
                # gdal translate to subset the input raster
                gdal.Translate(output_dir+f"/image_0"+str(count)+".tif", cbers4a_image, projWin = (xmin, ymax, xmax, ymin), xRes = res, yRes = -res)
                count += 1
            elif count >= 10000:
                # gdal translate to subset the input raster
                gdal.Translate(output_dir+f"/image_"+str(count)+".tif", cbers4a_image, projWin = (xmin, ymax, xmax, ymin), xRes = res, yRes = -res)
                count += 1
    # close the open dataset!!!
    cbers4a_image = None

def dataset_generator(imgs_dir, config, output_dir):
    imgs_dir = imgs_dir

    if config == 'config1':
        min_lvl = 0
        max_lvl = 1
        shadow_max_lvl = 1
        locality_degree = 2
    elif config == 'config2':
        min_lvl = 0
        max_lvl = 1
        shadow_max_lvl = 0
        locality_degree = 2
    elif config == 'config3':
        min_lvl = 0
        max_lvl = 0
        shadow_max_lvl = 1
        locality_degree = 2
    elif config == 'config4':
        min_lvl = 0
        max_lvl = 1
        shadow_max_lvl = 1
        locality_degree = 1
    elif config == 'config5':
        min_lvl = 0
        max_lvl = 1
        shadow_max_lvl = 1
        locality_degree = 3
    elif config == 'config6':
        min_lvl = 0
        max_lvl = 0.45
        shadow_max_lvl = 0
        locality_degree = 4
    elif config == 'config7':
        min_lvl = 0
        max_lvl = 0
        shadow_max_lvl = 1
        locality_degree = 4
    elif config == 'config8':
        min_lvl = 0
        max_lvl = 1
        shadow_max_lvl = 0
        locality_degree = 1
    elif config == 'config9':
        min_lvl = 0
        max_lvl = 0
        shadow_max_lvl = 0
        locality_degree = 1
    # elif config == 'config10':
    #     min_lvl = 0.0
    #     max_lvl = 0.0
    #     shadow_max_lvl = [0.4,0.5]
    #     locality_degree = 2
    # elif config == 'config11':
    #     min_lvl = 0.05
    #     max_lvl = 0.5
    #     shadow_max_lvl = 0
    #     locality_degree = 1
    
    
    for count in range(len(os.listdir(imgs_dir))):
        if count < 9:
            img_name = imgs_dir+f"/image_0000{count+1}.tif"
            img4bands = imageio.imread(img_name, format="GDAL")[:4, :, :]
        elif count >= 9 and count < 99:
            img_name = imgs_dir+f"/image_000{count+1}.tif"
            img4bands = imageio.imread(img_name, format="GDAL")[:4, :, :]
        elif count >= 99 and count < 999:
            img_name = imgs_dir+f"/image_00{count+1}.tif"
            img4bands = imageio.imread(img_name, format="GDAL")[:4, :, :]
        elif count >= 999:
            img_name = imgs_dir+f"/image_0{count+1}.tif"
            img4bands = imageio.imread(img_name, format="GDAL")[:4, :, :]
    
        N = max(np.unique(img4bands))
        img4bands = img4bands / N
        img4bands = torch.FloatTensor(img4bands).unsqueeze(0)
    
        out, cmask, smask = add_cloud_and_shadow(img4bands,
                                                    min_lvl=min_lvl,
                                                    max_lvl=max_lvl,
                                                    shadow_max_lvl=shadow_max_lvl,
                                                    const_scale=True,
                                                    decay_factor=1,
                                                    clear_threshold=[0.0,0.1],
                                                    locality_degree=locality_degree,
                                                    cloud_color=False,
                                                    channel_offset=2,
                                                    blur_scaling=0,
                                                    return_cloud=True
                                                    )
    
        # Convert the PyTorch tensor to a NumPy array
        out_array = out.numpy()
        out_array = np.squeeze(out_array, axis=0)
        out_array = (out_array*N)
    
        img_array_transp = out_array.transpose((1, 2, 0))
    
        red_img_transp = img_array_transp[:,:,0]
        green_img_transp = img_array_transp[:,:,1]
        blue_img_transp = img_array_transp[:,:,2]
        nir_img_transp = img_array_transp[:,:,3]
        # NDVI band
        ndvi_img = (nir_img_transp - red_img_transp) / (nir_img_transp + red_img_transp + 1e-6)
        ndvi_img = ndvi_img.astype('float32')
        # WI band
        M = (blue_img_transp + green_img_transp + red_img_transp) / 3 + 1e-6
        wi_img = (abs(red_img_transp - M) + abs(green_img_transp - M) + abs(blue_img_transp - M))/M
        wi_img = wi_img.astype('float32')
    
        img_array_final = np.dstack((img_array_transp,ndvi_img))
        img_array_final = np.dstack((img_array_final,wi_img))
    
        img_array_final = img_array_final.transpose((2, 0, 1))
    
        ds=gdal.Open(img_name)
    
        CreateGeoTiff(outRaster=output_dir+f'/imgs/6b_{config}_'+img_name.split('/')[-1], data=img_array_final,
                    geo_transform=ds.GetGeoTransform(), projection=ds.GetProjection(), dtype=gdal.GDT_Float32)            
        
        seg = segmentation_mask(cmask[0],
                            smask[0],
                            thin_range=(0.05,0.5))[0]
        seg_array = seg.numpy()
    
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = seg_array.shape
        geo_transform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        DataSet = driver.Create(output_dir+f'/masks/6b_{config}_mask_'+img_name.split('/')[-1][6:], cols, rows, 1, gdal.GDT_Byte)
        DataSet.SetGeoTransform(geo_transform)
        DataSet.SetProjection(projection)
    
        DataSet.GetRasterBand(1).WriteArray(seg_array)
        DataSet = None