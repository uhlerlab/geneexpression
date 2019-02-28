import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import visdom
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy.random as random
from scipy.io import loadmat
from collections import Counter
from PIL import Image
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# np.memmap 

def setup_data_loaders():
    countsInVitro = np.load('counts_matrix_in_vitro.npz', mmap_mode='r+')
    cellMetadataInVitro = np.loadtxt('cell_metadata_in_vitro.txt', skiprows=1, usecols=(0, 3))
    countsInVitroCscMatrix = csc_matrix((countsInVitro['data'], countsInVitro['indices'], countsInVitro['indptr']), shape=(130887, 25289)).toarray()
    countsInVitroPCA = IncrementalPCA(n_components=100, batch_size=100).fit_transform(countsInVitroCscMatrix)
    data_1 = []
    data_2 = []
    for i in range(len(countsInVitroPCA)):
        if int(cellMetadataInVitro[i][0]) >= 4:
            data_2.append(countsInVitroPCA[i])
        else:
            data_1.append(countsInVitroPCA[i])

    max_length = max(len(data_1), len(data_2))
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    idx_1 = np.random.permutation(len(data_1))
    idx_2 = np.random.permutation(len(data_2))

    # upsample smaller dataset
    if len(data_1)<max_length:
        idx_1 = np.random.randint(0, len(data_1), max_length)
    elif len(data_2)<max_length:
        idx_2 = np.random.randint(0, len(data_2), max_length)

    data_1 = data_1[idx_1]
    data_2 = data_2[idx_2]
#    data_len = min(len(data_1), len(data_2))
#    data_1 = data_1[0:data_len]
#    data_2 = data_2[0:data_len]
    print(data_1.shape, data_2.shape)
    data_1.dump('dat1')
    data_2.dump('dat2')

setup_data_loaders()
                                                                                                                                      53,1          Bot

