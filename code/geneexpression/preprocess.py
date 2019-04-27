import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import visdom
import random as rd
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy.random as random
from scipy.io import loadmat
from collections import Counter
from PIL import Image
from scipy.stats import wilcoxon
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import IncrementalPCA
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def setup_data_loaders():
    countsInVitro = np.load('counts_matrix_in_vitro.npz', mmap_mode='r+')
    cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
    clone_data = csc_matrix(
        (cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()
    countsInVitroCscMatrix = csc_matrix(
        (countsInVitro['data'], countsInVitro['indices'], countsInVitro['indptr']), shape=(130887, 25289)).toarray()
    metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
    cellMetadataInVitroType = np.genfromtxt(
        'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                          | (day4_6['Annotation'] == 'Monocyte')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    data_1 = [countsInVitroCscMatrix[i] for i in day2]
    data_2 = [countsInVitroCscMatrix[i] for i in day4_6]
    data_len = min(len(data_1), len(data_2))
    data_1 = data_1[0:data_len]
    data_2 = data_2[0:data_len]
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_1_hypothesis = []
    data_2_hypothesis = []
    # Hypothesis test for most differentially expressed genes
    for i in range(len(data_1[0])):
        data_1_gene = np.transpose(data_1)[i]
        data_2_gene = np.transpose(data_2)[i]
        stat, p = wilcoxon(data_1_gene, data_2_gene)
        alpha = 0.05
        if p <= alpha:
            data_1_hypothesis.append(np.transpose(data_1)[i])
            data_2_hypothesis.append(np.transpose(data_2)[i])
    data_1 = np.transpose(np.array(data_1_hypothesis))
    data_2 = np.transpose(np.array(data_2_hypothesis))
    pca = IncrementalPCA(n_components=100, batch_size=100)
    data_1 = pca.fit_transform(data_1)
    data_2 = pca.transform(data_2)
    print(data_1.shape, data_2.shape)
    data_1.dump('dat1')
    data_2.dump('dat2')


def setup_data_loaders_semisupervised():
    countsInVitro = np.load('counts_matrix_in_vitro.npz', mmap_mode='r+')
    cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
    clone_data = csc_matrix(
        (cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()
    countsInVitroCscMatrix = csc_matrix(
        (countsInVitro['data'], countsInVitro['indices'], countsInVitro['indptr']), shape=(130887, 25289)).toarray()
    metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
    cellMetadataInVitroType = np.genfromtxt(
        'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                          | (day4_6['Annotation'] == 'Monocyte')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    data_1 = []
    data_2 = []
    clone_df = pd.DataFrame(clone_data)
    day4_6Ind = []
    indices = []
    repeated = []
    for ind in day2[:500]:
        foundClone = False
        cloneInd = 0
        for i in range(len(clone_data[ind])):
            if clone_data[ind][i] == 1:
                foundClone = True
                cloneInd = i
        if foundClone != False:
            sameClone = pd.DataFrame(
                clone_df.loc[clone_df[cloneInd] == 1]).index
            sameCloneInd = [] 
            for i in range(len(day4_6)):
                if day4_6[i] in sameClone and day4_6[i] not in repeated:
                     repeated.append(day4_6[i])
                     sameCloneInd.append(day4_6[i]) 
            for elem in sameCloneInd:
                day4_6Ind.append(elem)
                data_1.append(countsInVitroCscMatrix[ind])
                data_2.append(countsInVitroCscMatrix[elem])
    remainingIndices = list(set(day4_6) - set(day4_6Ind))
    remainingDay2 = day2[500:]
    for i in range(len(remainingDay2)):
        data_1.append(countsInVitroCscMatrix[remainingDay2[i]])
        day4_6Ind.append(remainingIndices[i])
        data_2.append(countsInVitroCscMatrix[remainingIndices[i]])
    print(len(data_1), len(data_2))
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_1_hypothesis = []
    data_2_hypothesis = []
    for i in range(len(data_1[0])):
        data_1_gene = np.transpose(data_1)[i]
        data_2_gene = np.transpose(data_2)[i]
        stat, p = wilcoxon(data_1_gene, data_2_gene)
        alpha = 0.05
        if p <= alpha:
            data_1_hypothesis.append(np.transpose(data_1)[i])
            data_2_hypothesis.append(np.transpose(data_2)[i])
    data_1 = np.transpose(np.array(data_1_hypothesis))
    data_2 = np.transpose(np.array(data_2_hypothesis))
    pca = IncrementalPCA(n_components=100, batch_size=100)
    data_1 = pca.fit_transform(data_1)
    data_2 = pca.transform(data_2)
    data_1.dump('dat1')
    data_2.dump('dat2')
    np.array(day4_6Ind).dump('ind')


def setup_data_loaders_supervised():
    countsInVitro = np.load('counts_matrix_in_vitro.npz', mmap_mode='r+')
    cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
    clone_data = csc_matrix(
        (cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()
    countsInVitroCscMatrix = csc_matrix(
        (countsInVitro['data'], countsInVitro['indices'], countsInVitro['indptr']), shape=(130887, 25289)).toarray()
    metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
    cellMetadataInVitroType = np.genfromtxt(
        'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                          | (day4_6['Annotation'] == 'Monocyte')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    data_1 = []
    data_2 = []
    clone_df = pd.DataFrame(clone_data)
    repeated = []
    for ind in day2:
        foundClone = False
        cloneInd = 0
        for i in range(len(clone_data[ind])):
            if clone_data[ind][i] == 1:
                foundClone = True
                cloneInd = i
        if foundClone != False:
            sameClone = pd.DataFrame(
                clone_df.loc[clone_df[cloneInd] == 1]).index
            sameCloneInd = [] 
            for i in range(len(day4_6)):
                if day4_6[i] in sameClone and day4_6[i] not in repeated:
                     repeated.append(day4_6[i])
                     sameCloneInd.append(day4_6[i])
            for elem in sameCloneInd:
                data_1.append(countsInVitroCscMatrix[ind])
                data_2.append(countsInVitroCscMatrix[elem])
    data_1 = np.array(data_1)
    data_2 = np.array(data_2)
    data_1_hypothesis = []
    data_2_hypothesis = []
    for i in range(len(data_1[0])):
        data_1_gene = np.transpose(data_1)[i]
        data_2_gene = np.transpose(data_2)[i]
        stat, p = wilcoxon(data_1_gene, data_2_gene)
        alpha = 0.05
        if p <= alpha:
            data_1_hypothesis.append(np.transpose(data_1)[i])
            data_2_hypothesis.append(np.transpose(data_2)[i])
    data_1 = np.transpose(np.array(data_1_hypothesis))
    data_2 = np.transpose(np.array(data_2_hypothesis))
    pca = IncrementalPCA(n_components=100, batch_size=100)
    data_1 = pca.fit_transform(data_1)
    data_2 = pca.transform(data_2)
    data_1.dump('dat1')
    data_2.dump('dat2')


setup_data_loaders_semisupervised()
