import numpy as np
import torch
from numpy import linalg as LA
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

alpha = 0.05

def obtainInitialData():
    countsInVitro = np.load('counts_matrix_in_vitro.npz', mmap_mode='r+')
    cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
    clone_data = csc_matrix(
        (cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()

    countsInVitroCscMatrix = csc_matrix(
        (countsInVitro['data'], countsInVitro['indices'], countsInVitro['indptr']), shape=(130887, 25289)).toarray()
    metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6_neutrophil = day4_6.loc[(day4_6['Annotation'] == 'Neutrophil')]
    day4_6_monocyte = day4_6.loc[(day4_6['Annotation'] == 'Monocyte')]
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                          | (day4_6['Annotation'] == 'Monocyte')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    return day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data


def perform_wilcoxon(data_1, data_2, clone_data, day2, clusters=False):
    data_1, data_2 = np.array(data_1), np.array(data_2)
    data_1_hypothesis, data_2_hypothesis, cluster = [], [], []
    for i in range(data_1.shape[0]):
        data_1_gene = data_1[i]
        data_2_gene = data_2[i]
        stat, p = wilcoxon(data_1_gene, data_2_gene)
        if p <= alpha:
            data_1_hypothesis.append(np.transpose(data_1_gene))
            data_2_hypothesis.append(np.transpose(data_2_gene))
            if clusters:
                cluster.append(1*clone_data[day2[i]])
    data_1, data_2 = np.array(data_1_hypothesis), np.array(data_2_hypothesis) 
    return data_1, data_2, cluster


def PCA_and_normalize(data_1, data_2):
    pca = IncrementalPCA(n_components=100, batch_size=100)
    allElems = np.concatenate((preprocessing.normalize(
        data_1, norm='l2'), preprocessing.normalize(data_2, norm='l2')), axis=0)
    allElems = pca.fit_transform(allElems)
    data_1 = allElems[:len(data_1)]
    data_2 = allElems[len(data_1):]
    return data_1, data_2


def setup_data_loaders_unsupervised():
    day2, day4_6, _, _, countsInVitroCscMatrix, clone_data = obtainInitialData()
    data_1, data_2, day4_6Ind =  [], [], []
    randomIndices = rd.sample(range(0, len(day4_6)), len(day2))
    for i in range(len(day2)):
        data_1.append(countsInVitroCscMatrix[day2[i]])
        day4_6Ind.append(day4_6[randomIndices[i]])
        data_2.append(countsInVitroCscMatrix[day4_6[randomIndices[i]]])
    data_1, data_2, cluster = perform_wilcoxon(data_1, data_2, clone_data, day2, clusters=True)
    data_1, data_2 = PCA_and_normalize(data_1, data_2)
    data_1.dump('dat1')
    data_2.dump('dat2')
    cluster = np.array(cluster)
    print(cluster.shape, data_1.shape, data_2.shape)
    cluster.dump('clusters')
    np.array(day4_6Ind).dump('ind')


def setup_data_loaders_semisupervised(numPoints):
    day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data = obtainInitialData()
    randomIndices = rd.sample(range(0, len(day2)), numPoints)
    day46_to_day2, day4_6Ind, data_1, data_2 = {}, [], [], []
    for i in range(numPoints):
        clone_index = np.where(clone_data[day2[randomIndices[i]]] == 1)[0]
        try:
            neutrophils = day4_6_neutrophil.loc[np.where(
                clone_data[:, clone_index] == 1)[0]]
            n_count = neutrophils.count()[0]
        except:
            n_count = 0
        try:
            monocytes = day4_6_monocyte.loc[np.where(
                clone_data[:, clone_index] == 1)[0]]
            m_count = monocytes.count()[0]
        except:
            m_count = 0
        cell_index = neutrophils if n_count > m_count else monocytes
        for row in cell_index.iterrows():
            if row[0] not in day46_to_day2 and not np.isnan(row[1]['Time point']):
                day46_to_day2[row[0]] = day2[randomIndices[i]]
                day4_6Ind.append(row[0])
                data_1.append(countsInVitroCscMatrix[day2[randomIndices[i]]])
                data_2.append(countsInVitroCscMatrix[row[0]])
                break
    remainingday2 = list(set(day2) - set(day46_to_day2.values()))
    remainingday4_6 = list(set(day4_6) - set(day46_to_day2.keys()))
    randomIndicesday4 = rd.sample(
        range(0, len(remainingday4_6)), len(remainingday2))
    for i in range(len(randomIndicesday4)):
        data_1.append(countsInVitroCscMatrix[remainingday2[i]])
        day4_6Ind.append(remainingday4_6[randomIndicesday4[i]])
        data_2.append(
            countsInVitroCscMatrix[remainingday4_6[randomIndicesday4[i]]])
    data_1, data_2, _ = perform_wilcoxon(data_1, data_2, [], day2,  False)
    data_1, data_2 = PCA_and_normalize(data_1, data_2)
    print(data_1.shape, data_2.shape)
    data_1.dump('dat1')
    data_2.dump('dat2')
    np.array(day4_6Ind).dump('ind')


def setup_data_loaders_supervised():
    day2, day4_6, day4_6_neutrophil, day4_6_monocyte, countsInVitroCscMatrix, clone_data = obtainInitialData()
    data_1, data_2, day4_6Ind, day46_to_day2 = [], [], [], {}
    for ind in day2:
        clone_index = np.where(clone_data[ind] == 1)[0]
        try:
            neutrophils = day4_6_neutrophil.loc[np.where(
                clone_data[:, clone_index] == 1)[0]]
            n_count = neutrophils.count()[0]
        except:
            n_count = 0
        try:
            monocytes = day4_6_monocyte.loc[np.where(
                clone_data[:, clone_index] == 1)[0]]
            m_count = monocytes.count()[0]
        except:
            m_count = 0
        cell_index = neutrophils if n_count > m_count else monocytes
        for row in cell_index.iterrows():
            if row[0] not in day46_to_day2 and not np.isnan(row[1]['Time point']):
                day46_to_day2[row[0]] = ind
                day4_6Ind.append(row[0])
                data_1.append(countsInVitroCscMatrix[ind])
                data_2.append(countsInVitroCscMatrix[row[0]])
                break
    data_1, data_2, _ = perform_wilcoxon(data_1, data_2, [], day2, day4_6Ind, clusters=False)
    data_1, data_2 = PCA_and_normalize(data_1, data_2)
    print(data_1.shape, data_2.shape)
    data_1.dump('dat1')
    data_2.dump('dat2')
    np.array(day4_6Ind).dump('ind')


def wot_preprocess():
    countsInVitro = np.load('counts_matrix_in_vitro.npz', mmap_mode='r+')
    cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
    clone_data = csc_matrix(
        (cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()
    countsInVitroCscMatrix = csc_matrix(
        (countsInVitro['data'], countsInVitro['indices'], countsInVitro['indptr']), shape=(130887, 25289)).toarray()
    metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
    day4_6 = metadata.loc[metadata['Time point'] > 3]
    day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                          | (day4_6['Annotation'] == 'Monocyte')]
    clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
    cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
    clone_metadata = metadata.loc[cell_index]
    day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()
    cell_day = []
    cell_id = []
    for i in range(len(day2)):
        cell_id.append("cell_" + str(day2[i]))
        cell_day.append(0.0)
    for i in range(len(day4_6)):
        cell_id.append("cell_" + str(day4_6[i]))
        cell_day.append(1.0)
    cell_day = np.array(cell_day)
    cell_id = np.array(cell_id)
    cell_day_pd = pd.DataFrame({'id': cell_id, 'day': cell_day})
    cell_day_pd.index = cell_day_pd.index.astype(str, copy=False)
    cell_day_pd = cell_day_pd.astype(str, copy=False)
    np.savetxt("cell_day.txt", cell_day_pd.values,
               fmt='%s', header="id,day", delimiter=',')
    geneexpression = []
    for i in range(len(cell_id)):
        nextIndex = int(cell_id[i][5:])
        nextMatrix = countsInVitroCscMatrix[nextIndex]
        geneexpression.append(nextMatrix)
    pca = IncrementalPCA(n_components=100, batch_size=100)
    geneexpression = pca.fit_transform(np.array(geneexpression))
    header = []
    for i in range(100):
        header.append(i/1.0)
    header_str = "id"
    for i in range(len(header)):
        header_str += ","
        header_str += str(header[i])
    geneexpression_pd = pd.DataFrame({'id': cell_id})
    for i in range(len(header)):
        geneexpression_pd[i] = geneexpression[:, i]
    geneexpression_pd.index = geneexpression_pd.index.astype(str, copy=False)
    geneexpression_pd = geneexpression_pd.astype(str, copy=False)
    np.savetxt("geneexpression.txt", geneexpression_pd.values,
               fmt='%s', header=header_str, delimiter=',')


setup_data_loaders_semisupervised(500)
