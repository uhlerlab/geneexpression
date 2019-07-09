import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
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
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def own_dataset():
    data_1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 3000)
    data_2 = np.random.multivariate_normal([0, 0], [[10, 0], [0, 10]], 3000)
    # print(len(data_1[0]))
    data_1.dump('dat1')
    data_2.dump('dat2')

def setup_data_loaders_unsupervised():
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
    clusters = []
    for i in range(len(day2)):
        #print(1*clone_data[day2[i]])
        clusters.append(1*clone_data[day2[i]])
        data_1.append(countsInVitroCscMatrix[day2[i]])
        #day4_6Ind.append(day4_6[i])
        data_2.append(countsInVitroCscMatrix[day4_6[i]])
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
    allElems = np.concatenate((preprocessing.normalize(data_1, norm='l2'), preprocessing.normalize(data_2, norm='l2')), axis=0)
    allElems = pca.fit_transform(allElems)
    data_1 = allElems[:len(data_1)]
    data_2 = allElems[len(data_1):]
    data_1.dump('dat1')
    data_2.dump('dat2')
    clusters = np.array(clusters)
    clusters.dump('clusters')
    print(data_1.shape, data_2.shape, clusters.shape)
    #np.array(day4_6Ind).dump('ind')


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
    num = 0
    index = 0
    remainingDay2 = []
    while num < 750:
        ind = day2[index]
        foundClone = False
        cloneInd = 0
        for i in range(len(clone_data[ind])):
            if clone_data[ind][i] == 1:
                foundClone = True
                cloneInd = i
                break
        if foundClone != False:
            sameClone = pd.DataFrame(
                clone_df.loc[clone_df[cloneInd] == 1]).index
            elem = -1
            for i in range(len(day4_6)):
                if day4_6[i] in sameClone and day4_6[i] not in day4_6Ind:
                    elem = day4_6[i]
                    break
            if elem != -1:
                remainingDay2.append(ind)
                day4_6Ind.append(elem)
                data_1.append(countsInVitroCscMatrix[ind])
                data_2.append(countsInVitroCscMatrix[elem])
                num += 1
        index += 1
    remainingIndices = list(set(day4_6) - set(day4_6Ind))
    remainingDay2 = list(set(day2) - set(remainingDay2))
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
    allElems = np.concatenate((preprocessing.normalize(data_1, norm='l2'), preprocessing.normalize(data_2, norm='l2')), axis=0)
    allElems = pca.fit_transform(allElems)
    data_1 = allElems[:len(data_1)]
    data_2 = allElems[len(data_1):]
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
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    data_1 = []
    data_2 = []
    day4_6Ind = []
    clone_df = pd.DataFrame(clone_data)
    for ind in day2:
        foundClone = False
        cloneInd = 0
        for i in range(len(clone_data[ind])):
            if clone_data[ind][i] == 1:
                foundClone = True
                cloneInd = i
                break
        if foundClone != False:
            sameClone = pd.DataFrame(
                clone_df.loc[clone_df[cloneInd] == 1]).index
            elem = -1
            for i in range(len(day4_6)):
                if day4_6[i] in sameClone and day4_6[i] not in day4_6Ind:
                    elem = day4_6[i]
                    break
            if elem != -1:
                day4_6Ind.append(elem)
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
    #print(data_1.shape, data_2.shape)
    pca = IncrementalPCA(n_components=100, batch_size=100)
    allElems = np.concatenate((preprocessing.normalize(data_1, norm='l2'), preprocessing.normalize(data_2, norm='l2')), axis=0)
    allElems = pca.fit_transform(allElems)
    data_1 = allElems[:len(data_1)]
    data_2 = allElems[len(data_1):]
    data_1.dump('dat1')
    data_2.dump('dat2')
    np.array(day4_6Ind).dump('ind')

# own_dataset()
setup_data_loaders_unsupervised()
