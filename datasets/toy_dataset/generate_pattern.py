import numpy as np
import matplotlib.pyplot as plt

DIR = "./data/"
NAME = "checker"
N_SAMPLES = 2000

# generate source data, uniformly [0, 1]
data1 = np.random.rand(N_SAMPLES, 2)
#plt.scatter(data1[:,0], data1[:,1])

# generate target data
data2 = []
data2.append(np.random.rand(int(N_SAMPLES/5), 2))
data2.append(np.random.rand(int(N_SAMPLES/5*2),2)/2)
data2.append(np.random.rand(int(N_SAMPLES/5*2),2)/2+0.5)
data2 = np.concatenate(data2)
np.random.shuffle(data2)
#plt.scatter(data2[:,0], data2[:,1])

print(len(data1), len(data2))

#save data
np.save(DIR+NAME+"_source", data1)
np.save(DIR+NAME+"_target", data2)
