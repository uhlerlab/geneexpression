from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import IncrementalPCA

countsInVitroPCA = np.load('countsinVitroPCA')
cellMetadataInVitroDay = np.loadtxt(
    'cell_metadata_in_vitro.txt', skiprows=1, usecols=(0,))
cellMetadataInVitroType = np.genfromtxt(
    'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
transportedPoints = np.loadtxt('results/exp4/nehaenv230_trans.txt')
targetDataset = []
y = []
transportedActual = []
#cellMetadataInVitroType = cellMetadataInVitroType.tostring().decode("ascii")
#print(len(countsInVitroPCA))
for i in range(102638):
   # print(cellMetadataInVitroType[i])
    if int(cellMetadataInVitroDay[i]) >= 4 and cellMetadataInVitroType[i] == 'Monocyte':
        targetDataset.append(countsInVitroPCA[i])
        y.append(0)
        transportedActual.append(transportedPoints[i])
    if int(cellMetadataInVitroDay[i]) >= 4 and cellMetadataInVitroType[i] == 'Neutrophil':
        targetDataset.append(countsInVitroPCA[i])
        y.append(1)
        transportedActual.append(transportedPoints[i])
print(len(targetDataset))
trainDataset = targetDataset[:26250]
testDataset = targetDataset[26250:]
lr = LogisticRegression(multi_class='ovr').fit(trainDataset, y[:26250])
# training accuracy
trainingPred = lr.predict(trainDataset)
totalTraining = 0
for i in range(26250):
    #print(i, len(y), len(trainingPred))
    if y[i] == trainingPred[i]:
        totalTraining += 1
print("Training accuracy: ", totalTraining/26250.0)

# test accuracy
testPred = lr.predict(testDataset)
totalTest = 0
for i in range(26250, 29167):
    if y[i] == testPred[i - 26250]:
        totalTest += 1
print("Test accuracy: ", totalTest/2917.0)

# transportedAccuracy
predictedY = lr.predict(transportedActual)
totalCount = 0
for i in range(len(y)):
    if y[i] == predictedY[i]:
        totalCount += 1
print("Transported Accuracy: ", totalCount/len(predictedY))
