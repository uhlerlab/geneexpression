from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
import random
import pandas as pd
import matplotlib
from scipy.sparse import csr_matrix, csc_matrix
matplotlib.use('agg')
import matplotlib.pyplot as plt
#retrieve points
transportedPoints = np.loadtxt('results/exp4/nehaenv2680_trans.txt')
day2 = np.load('dat1')
day4through6 = np.load('dat2')
print(len(transportedPoints), len(day2), len(day4through6))
#downsample to 500 points
numPoints = 500
randomIndices = random.sample(range(0, len(day4through6)), numPoints)
transportedPoints500 = [transportedPoints[ind] for ind in randomIndices]
day2500 = [day2[ind] for ind in randomIndices]
day4through6500 = [day4through6[ind] for ind in randomIndices]

#use tsne to transform 
tsne = TSNE(n_components=2)
allElems = day2500[:]
allElems.extend(day4through6500[:])
allElems.extend(transportedPoints500[:])
print(len(allElems), len(allElems[0]))
allElems = tsne.fit_transform(allElems)
day2TSNE = allElems[:numPoints]
day4through6TSNE = allElems[numPoints:2*numPoints]
transportedPointsTSNE = allElems[2*numPoints:]
print(len(day2TSNE), len(day2TSNE[0]))
:
#plot resulting points
plt.scatter(day2TSNE[:,0], day2TSNE[:,1], color='r', marker='o')
plt.scatter(day4through6TSNE[:,0], day4through6TSNE[:,1], color='b', marker='o')
plt.scatter(transportedPointsTSNE[:,0], transportedPointsTSNE[:,1], color='g', marker='o')
plt.xlabel('x')
plt.ylabel('y')
#plt.show()
plt.legend(['day2', 'day4through6', 'transportedPoints'])
plt.savefig('tsneplot.png')

