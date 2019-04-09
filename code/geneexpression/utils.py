import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import visdom
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy.random as random
from scipy.io import loadmat
from collections import Counter
from PIL import Image
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
random.seed(0)


# ============= ARGUMENT PARSING ==============

def setup_args():

    options = argparse.ArgumentParser()

    #setup
    options.add_argument('-sd', action="store", dest="save_file", default="./results/")
    # options.add_argument('-dat1', action="store", dest="source_data_file", default="../../../datasets/zebrafish/ZFDOME_WT.npy")
    # options.add_argument('-dat2', action="store", dest="target_data_file", default="../../../datasets/zebrafish/ZF50_WT.npy")
    options.add_argument('-env', action="store", dest="env", default="ubOT_ZF")

    # training arguments
    options.add_argument('-bs', action="store", dest="batch_size", default = 256, type = int)
    options.add_argument('-iter', action="store", dest="max_iter", default = 1000, type = int)
    options.add_argument('-citer', action="store", dest="critic_iter", default = 2, type = int)
    options.add_argument('-lambG', action="store", dest="lambG", default=10, type = float)
    options.add_argument('-lambG2', action="store", dest="lambG2", default=1, type = float)
    options.add_argument('-lamb0', action="store", dest="lamb0", default=1, type = float)
    options.add_argument('-lamb1', action="store", dest="lamb1", default=10, type = float)
    options.add_argument('-lamb2', action="store", dest="lamb2", default=100, type = float)
    options.add_argument('-psi', action="store", dest="psi2", default="JS")
    options.add_argument('-lrG', action="store", dest="lrG", default=1e-4, type = float)
    options.add_argument('-lrD', action="store", dest="lrD", default=1e-4, type = float)

    #options.add_argument('-psi', action="store", dest="psi2", default="EQ")

    # model arguments
    options.add_argument('-nz', action="store", dest="nz", default=100, type = int)
    options.add_argument('-nh', action="store", dest="n_hidden", default=500, type = int)

    return options.parse_args()

#============= DATA LOADING ==============

def setup_data_loaders(batch_size):
    data_1 = np.load('dat1')
    data_2 = np.load('dat2')
    print(data_1.shape, data_2.shape)

    dataset = TensorDataset(torch.from_numpy(data_1).float(), torch.from_numpy(data_2).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return dataloader

def setup_data_loaders_supervised(batch_size):
    data_1_1 = np.load('dat1_1')
    data_2_1 = np.load('dat2_1')
    supervised = np.load('supervised')
    dataset1 = TensorDataset(torch.from_numpy(data_1_1).float(), torch.from_numpy(data_2_1).float())
    dataloader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, pin_memory=True)

    dataset2 = TensorDataset(torch.from_numpy(supervised).float())
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False, pin_memory=True)

    return dataloader1, dataloader2

#============= DATA LOGGING ==============



class Tracker(object):
    def __init__(self):
        self.epoch = 0
        self.sum = 0
        self.count = 0

        self.list_avg = []

    def add(self, val, count=1):
        self.sum += val*count
        self.count += count

    def tick(self):
        avg = self.sum / self.count
        self.list_avg.append(avg)

        self.epoch += 1
        self.sum = 0
        self.count = 0

    def get_status(self):
        return self.list_avg


# ============= DATA VISUALIZATION ==============

def init_visdom(env):
    vis = visdom.Visdom(port='8096')
    # vis.port(8096)
    # vis.close(env=env)
    return vis


def plot(tracker, epoch, s_inputs, s_generated, t_inputs, env, vis):
    # close old plots
    vis.close(env=env)
    # update plots
    print("got here", s_inputs)
    W1Plot = vis.line(
        Y=np.array(tracker.get_status()),
        X=np.array(range(tracker.epoch)),
        env=env,
        name="Loss",
    )
    
   # Scatter = vis.scatter(
        #X=s_inputs,
        #env=env,
        #name="Source",
        #opts=dict(
           # markercolor=s_inputs
        #),
   # )

    #Scatter = vis.scatter(
        #X=t_inputs,
        #env=env,
        #name="Target",
        #opts=dict(
        #    markercolor=np.zeros(shape=(len(t_inputs.data),)),
        #    markersymbol="cross",
        #),
        #update='add'
#    )

    # Scatter = vis.scatter(
       # X=s_generated,
       # win=Scatter,
       # env=env,
       # name="Transported",
#        opts=dict(
#            markercolor=s_inputs[:,0]+10
#        ),
       # update='add'
    # )

