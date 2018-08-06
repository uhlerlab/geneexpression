import numpy as np
import torch
import visdom
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset

#============= ARGUMENT PARSING ==============

def setup_args():

    options = argparse.ArgumentParser()

    #setup
    options.add_argument('-sd', action="store", dest="save_file", default="./results/")
    options.add_argument('-dat1', action="store", dest="source_data_file", default="../../../datasets/toy_dataset/data/mix_gauss_source.npy")
    options.add_argument('-dat2', action="store", dest="target_data_file", default="../../../datasets/toy_dataset/data/mix_gauss_target.npy")
    options.add_argument('-env', action="store", dest="env", default="OT_unbalanced_toy")
    #options.add_argument('-vis', action="store", dest="use_visdom", default=True, type = bool)

    # training arguments
    options.add_argument('-bs', action="store", dest="batch_size", default = 64, type = int)
    options.add_argument('-iter', action="store", dest="max_iter", default = 1000, type = int)
    options.add_argument('-citer', action="store", dest="critic_iter", default = 5, type = int)
    options.add_argument('-lamb1', action="store", dest="lamb1", default=1, type = float)
    options.add_argument('-lamb2', action="store", dest="lamb2", default=10, type = float)

    # model arguments
    options.add_argument('-nz', action="store", dest="nz", default=2, type = int)
    options.add_argument('-nh', action="store", dest="n_hidden", default=512, type = int)
    
    return options.parse_args()

#============= DATA LOADING ==============

def setup_data_loaders(batch_size, data_file_1, data_file_2):
    data_1 = np.load(data_file_1)
    data_2 = np.load(data_file_2)

    data_len = min(len(data_1), len(data_2))
    data_1 = data_1[0:data_len]
    data_2 = data_2[0:data_len]

    print(data_1.shape, data_2.shape)

    dataset = TensorDataset(torch.from_numpy(data_1).float(), torch.from_numpy(data_2).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return dataloader

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


#============= DATA VISUALIZATION ==============


def init_visdom(env):
    vis = visdom.Visdom()
    vis.close(env=env)


def plot(tracker, s_inputs, s_generated, t_inputs, env)
    # close old plots
    vis.close(env=env)
    # update plots
    W1Plot=vis.line(
        Y=np.array(tracker.get_status()),
        X=np.array(range(tracker.epoch)),
        env=env,
        name="Loss",

    )

    Scatter = vis.scatter(
        X=s_inputs.cpu().data.numpy(),
        #win=Scatter,
        env=env,
        name="Source",
        opts=dict(
            markercolor=s_inputs.cpu().data.numpy()[:,1]
        ),
    )

    Scatter = vis.scatter(
        X=t_inputs.cpu().data.numpy(),
        win=Scatter,
        env=env,
        name="Target",
        opts=dict(
            markercolor=np.zeros(shape=(len(t_inputs.data),)),
            markersymbol="cross",
        ),
        update='add'
    )

    Scatter = vis.scatter(
        X=s_generated.cpu().data.numpy(),
        win=Scatter,
        env=env,
        name="Transported",
        opts=dict(
            markercolor=s_inputs.cpu().data.numpy()[:,1]
        ),
        update='add'
    )

