import numpy as np
from osgeo import gdal
import imageio.v2 as imageio
import itertools
import time
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

# Caminhos das imagens e máscaras
img_dirs = [
    '/media/reginaldo/pfc-dados/dataset-principal/dados-teste/amazonas/testes/img-final-23-05.tif',
    '/media/reginaldo/pfc-dados/dataset-principal/dados-teste/amazonas/testes/img-final-23-06.tif',
    '/media/reginaldo/pfc-dados/dataset-principal/dados-teste/amazonas/testes/img-final-23-07.tif',
    '/media/reginaldo/pfc-dados/dataset-principal/dados-teste/amazonas/testes/img-final-23-08.tif'
]

mask_dirs = [
    '/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-teste/amazonas/mascara-20230527.tif',
    '/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-teste/amazonas/mascara-20230627.tif',
    '/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-teste/amazonas/mascara-20230728.tif',
    '/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/resultados-teste/amazonas/mascara-20230828.tif'
]

inicio_leitura = time.time()

# Carregar todas as imagens e máscaras
imgs = [imageio.imread(img_dir) for img_dir in img_dirs]
masks = [imageio.imread(mask_dir) for mask_dir in mask_dirs]

fim_leitura = time.time()
print(f'Tempo de leitura das imagens: {fim_leitura - inicio_leitura} seg.')

# Caminho de saída
output_base_dir = '/media/reginaldo/pfc-dados/dataset-principal/modelos-treinados/treino1/mosaicos/amazonas/mosaicos-com-2cenas'

# Gerar todas as permutações das 2, 3 ou 4 imagens e máscaras
elements = [2, 3]
permutations = list(itertools.permutations(elements))

inicio_iteracao = time.time()

for perm in permutations:
    # Criar o mosaico
    mosaico = np.zeros((imgs[0].shape[0], imgs[0].shape[1], 3))
    # Nome do arquivo com base na permutação atual
    output_name = f'mosaico{"".join([str(p + 1) for p in perm])}-amazonas.tif'
    output_path = os.path.join(output_base_dir, output_name)

    for x in range(imgs[0].shape[0]):
        for y in range(imgs[0].shape[1]):
            for idx in perm:
                # Aplicar a máscara para cada imagem na ordem da permutação
                if masks[idx][x, y] == 0:
                    mosaico[x, y, :3] = imgs[idx][x, y, :3]
                    break
            else:
                # Se nenhuma máscara foi aplicada, usa a última imagem da permutação
                mosaico[x, y, :3] = imgs[perm[0]][x, y, :3]

    mosaico = mosaico.transpose((2, 0, 1))

    # Obter a referência geográfica da primeira imagem da permutação
    ds = gdal.Open(img_dirs[perm[0]])
    CreateGeoTiff(
        outRaster=output_path,
        data=mosaico,
        geo_transform=ds.GetGeoTransform(),
        projection=ds.GetProjection(),
        dtype=gdal.GDT_Float32
    )

    print(f'Mosaico gerado: {output_path}')

fim_iteracao = time.time()
print(f'Tempo total da iteração: {fim_iteracao - inicio_iteracao} seg.')
