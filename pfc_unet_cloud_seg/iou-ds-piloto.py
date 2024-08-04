import imageio.v2 as imageio
import numpy as np
from sklearn.metrics import jaccard_score
#from sklearn.multioutput import MultiOutputClassifier
'''
# máscara real
real_dir = '/media/reginaldo/pfc-dados/dataset-piloto/mascara-real.tif'
real = imageio.imread(real_dir)
real = np.asarray(real)

real_list = [i[0] for i in real]

# Treino 1
treino1_dir = '/media/reginaldo/pfc-dados/dataset-piloto/treino1/mascara-modelo-treino1.tif'
treino1 = imageio.imread(treino1_dir)
treino1 = np.asarray(treino1)
treino1_list = [j[0] for j in treino1]
j1 = jaccard_score(real_list,treino1_list, average=None)

# Treino 2
treino2_dir = '/media/reginaldo/pfc-dados/dataset-piloto/treino2/mascara-modelo-treino2.tif'
treino2 = imageio.imread(treino2_dir)
treino2 = np.asarray(treino2)
treino2_list = [j[0] for j in treino2]
j2 = jaccard_score(real_list,treino2_list, average=None)

# Treino 3
treino3_dir = '/media/reginaldo/pfc-dados/dataset-piloto/treino3/mascara-modelo-treino3.tif'
treino3 = imageio.imread(treino3_dir)
treino3 = np.asarray(treino3)
treino3_list = [j[0] for j in treino3]
j3 = jaccard_score(real_list,treino3_list, average=None)

# Treino 4
treino4_dir = '/media/reginaldo/pfc-dados/dataset-piloto/treino4/mascara-modelo-treino4.tif'
treino4 = imageio.imread(treino4_dir)
treino4 = np.asarray(treino4)
treino4_list = [j[0] for j in treino4]
j4 = jaccard_score(real_list,treino4_list, average=None)

# Treino 5
treino5_dir = '/media/reginaldo/pfc-dados/dataset-piloto/treino5/mascara-modelo-treino5.tif'
treino5 = imageio.imread(treino5_dir)
treino5 = np.asarray(treino5)
treino5_list = [j[0] for j in treino5]
j5 = jaccard_score(real_list,treino5_list, average=None)

print(f'Treino 1: {j1*1e2}\nTreino 2: {j2*1e2}\nTreino 3: {j3*1e2}\nTreino 4: {j4*1e2}\nTreino 5: {j5*1e2}')
'''
classes = ['fundo', 'nuvem-densa','nuvem-fina','sombra']
numeracao = ['1','2','3','4','5']
for num in numeracao:
    print('\n')
    for classe in classes:
        if num == '5' and classe == 'nuvem-densa':
            print(f'Treino {num} ({classe}): 0.00 %')
        else:
            img_real = f'nuvens-reais/separated-classes/{classe}-real.tif'
            img_real = imageio.imread(img_real)

            img_treino = f'nuvens-reais/separated-classes/{classe}-treino{num}.tif'
            img_treino = imageio.imread(img_treino)

            intersection = np.logical_and(img_real, img_treino)
            union = np.logical_or(img_real, img_treino)
            print(f'Treino {num} ({classe}): {round(intersection.sum() / union.sum(),5)*1e2} %')
        

