from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.manifold import TSNE
import random
import pandas as pd
import matplotlib
from scipy.sparse import csr_matrix, csc_matrix
matplotlib.use('agg')
import matplotlib.pyplot as plt
#retrieve points
transportedPoints = np.loadtxt('results/exp4/nehaenv990_trans.txt')
day2 = np.load('dat1')
day4through6 = np.load('dat2')

#downsample to 500 points
randomIndices = random.sample(range(0, 1527), 500)
transportedPoints500 = [transportedPoints[ind] for ind in randomIndices]
day2500 = [day2[ind] for ind in randomIndices]
day4through6500 = [day4through6[ind] for ind in randomIndices]

#use tsne to transform 
tsne = TSNE(n_components=2)
day2TSNE = tsne.fit_transform(day2500)
transportedPointsTSNE = tsne.fit_transform(transportedPoints500)
day4through6TSNE = tsne.fit_transform(day4through6500)

#print(day2TSNE)
#print(transportedPointsTSNE)
#print(day4through6TSNE)

#plot resulting points
plt.plot(day2TSNE[:,0], day2TSNE[:,1], color='r', marker='o')
plt.plot(transportedPointsTSNE[:,0], transportedPointsTSNE[:,1], color='g', marker='o')
plt.plot(day4through6TSNE[:,0], day4through6TSNE[:,1], color='b', marker='o')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()
plt.legend(['day2', 'day4through6', 'transportedPoints'])
plt.savefig('tsneplot.png')


# retrieve points
'''
cloneAnnotation = np.load('clone_annotation_in_vitro.npz')
clone_data = csc_matrix((cloneAnnotation['data'], cloneAnnotation['indices'], cloneAnnotation['indptr']), shape=(130887, 5864)).toarray()
transportedPoints = np.loadtxt('results/exp4/nehaenv990_trans.txt')
data_2 = np.load('dat2')
celllMetadataInVitroDay = np.loadtxt(
    'cell_metadata_in_vitro.txt', skiprows=1, usecols=(0,))
cellMetadataInVitroType = np.genfromtxt(
    'cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
metadata = pd.read_csv('cell_metadata_in_vitro.txt', sep='\\t', header=0)
cellMetadataInVitroType = np.genfromtxt('cell_metadata_in_vitro.txt', dtype='str',  skip_header=1, usecols=(2,))
day4_6 = metadata.loc[metadata['Time point'] > 3]
day4_6 = day4_6.index[(day4_6['Annotation'] == 'Neutrophil')
                        | (day4_6['Annotation'] == 'Monocyte')]
clone_index = np.reshape(
        np.array(np.sum(clone_data[day4_6], axis=0)), 5864) > 0
cell_index = np.reshape(
        np.array(np.sum(clone_data[:, clone_index], axis=1)), 130887) > 0
clone_metadata = metadata.loc[cell_index]
day2 = clone_metadata.index[clone_metadata['Time point'] < 3].tolist()

randomIndices = random.sample(range(0, len(day2)), 500)
yday4through6 = []
for i in randomIndices:
	if cellMetadataInVitroType[day4_6[i]] == 'Monocyte':
	    yday4through6.append(0)
	if cellMetadataInVitroType[day4_6[i]] == 'Neutrophil':
	    yday4through6.append(1)
transportedPoints500 = [transportedPoints[i] for i in randomIndices]
day4through6500 = [data_2[i] for i in randomIndices]

clf=LinearDiscriminantAnalysis(n_components=2)
transportedPointsLDA=clf.fit_transform(transportedPoints500, yday4through6)
day4through6LDA=clf.transform(day4through6500)

print(transportedPointsLDA)
print(day4through6LDA)
plt.plot(transportedPointsLDA[:, 0], transportedPointsLDA[:, 1], linestyle='None', color='g', marker='o')
plt.plot(day4through6LDA[:, 0], day4through6LDA[:, 1],
         color='b', linestyle='None', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['day2', 'day4through6', 'transportedPoints'])
plt.savefig('ldaplot.png')
'''
