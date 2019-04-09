from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import IncrementalPCA
import pandas as pd

cellMetadataInVitroDay = np.loadtxt(
    'cell_metadata_in_vitro.txt', skiprows=1, usecols=(0,))
cellMetadataInVitroType = np.genfromtxt(
    'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
cellMetadataInVitroType = np.genfromtxt('cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
day4_6 = metadata.loc[metadata['Time point'] > 3]
day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                        | (day4_6['Annotation'] == 'Monocyte')]
transportedPoints = np.loadtxt('results/exp4/nehaenv990_trans.txt')
targetDataset = np.load('dat2')
y = []
for i in range(len(transportedPoints)):
    if cellMetadataInVitroType[day4_6[i]] == 'Monocyte':
        y.append(0)
       
    if cellMetadataInVitroType[day4_6[i]] == 'Neutrophil':
        y.append(1)
       
print(len(targetDataset))
point9 = int(len(targetDataset)*0.9)

trainDataset = targetDataset[:point9]
testDataset = targetDataset[point9:]
lr = LogisticRegression(multi_class='ovr').fit(trainDataset, y[:point9])
# training accuracy
trainingPred = lr.predict(trainDataset)
totalTraining = 0
for i in range(point9):
    #print(i, len(y), len(trainingPred))
    if y[i] == trainingPred[i]:
        totalTraining += 1
print("Training accuracy: ", totalTraining/point9)

# test accuracy
testPred = lr.predict(testDataset)
totalTest = 0
for i in range(point9, len(trainDataset)):
    if y[i] == testPred[i - point9]:
        totalTest += 1
print("Test accuracy: ", totalTest/(len(targetDataset) - point9))

# transportedAccuracy
predictedY = lr.predict(transportedPoints)
totalCount = 0
for i in range(len(y)):
   # print(predictedY[i])
    if y[i] == predictedY[i]:
        totalCount += 1
print("Transported Accuracy: ", totalCount/len(predictedY))
