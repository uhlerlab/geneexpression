import numpy as np
import matplotlib.pyplot as plt

# arguments
DIR = "./data/"
NAME = "mix_gauss"
SIZE = 1000
SPLIT = np.array([0.1, 0.3, 0.6])

MEANS = np.array([0.,3.,6.])
SCALES = np.array([.5, .5, .5])
SHIFT = 3

n_samples = np.cumsum(SIZE*SPLIT).astype(int)
print(n_samples)

# source data
data1 = []

for m,s,n in zip(MEANS, SCALES, n_samples):
    data1.append(np.random.normal(loc=[m, 0], scale=[s,s], size=[n, 2]))
    
data1 = np.concatenate(data1)
np.random.shuffle(data1)

# target data
data2 = []

for m,s,n in zip(MEANS, SCALES, n_samples[::-1]):
    data2.append(np.random.normal(loc=[m, SHIFT], scale=[s,s], size=[n, 2]))
    
data2 = np.concatenate(data2)
np.random.shuffle(data2)

np.save(DIR+NAME+"_source", data1)
np.save(DIR+NAME+"_target", data2)