import numpy as np
from sklearn.manifold import TSNE
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

#retrieve points
transportedPoints = np.loadtxt('results/exp4/nehaenv230_trans.txt')
day2 = np.load('dat1')
day4through6 = np.load('dat2')

#downsample to 500 points
randomIndices = random.sample(range(0, 102637), 500)
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
