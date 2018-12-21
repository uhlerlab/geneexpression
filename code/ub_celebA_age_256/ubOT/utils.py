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

random.seed(0)

#============= ARGUMENT PARSING ==============

def setup_args():

    options = argparse.ArgumentParser()

    #setup
    options.add_argument('-sd', action="store", dest="save_file", default="./results/")
    options.add_argument('-pt', action="store", dest="pretrained_file", default="./pretrained/CelebA_new_256_0.000001_2_49.pth")
    options.add_argument('-dat1', action="store", dest="source_data_file", default="../../../datasets/celebA_latents_256/CelebA_256_0.000001_latents_age_1.npy")
    options.add_argument('-dat2', action="store", dest="target_data_file", default="../../../datasets/celebA_latents_256/CelebA_256_0.000001_latents_age_0.npy")
    options.add_argument('-env', action="store", dest="env", default="ubOT_celebA")

    # training arguments
    options.add_argument('-bs', action="store", dest="batch_size", default = 128, type = int)
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
    options.add_argument('-nz', action="store", dest="nz", default=256, type = int)
    options.add_argument('-nh', action="store", dest="n_hidden", default=256, type = int)

    return options.parse_args()

#============= DATA LOADING ==============

def setup_data_loaders(batch_size, data_file_1, data_file_2):
    data_1 = np.load(data_file_1)
    data_2 = np.load(data_file_2)

    idx_1 = np.random.permutation(len(data_1))
    idx_2 = np.random.permutation(len(data_2))

    data_1 = data_1[idx_1]
    data_2 = data_2[idx_2]

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
    return vis


def plot(tracker, tracker_plot, scale_plot, s_inputs, s_generated, s_scale, t_inputs, env, vis):
    # update plots

    if tracker_plot is None:
        W1Plot=vis.line(
            Y=np.array(tracker.get_status()),
            X=np.array(range(tracker.epoch)),
            env=env,
        )
    else:
        W1Plot=vis.line(
            Y=np.array(tracker.get_status()),
            X=np.array(range(tracker.epoch)),
            env=env,
            win=tracker_plot,
            update='replace',
        )

    if scale_plot is None:
        scale_plot=vis.boxplot(
            X=s_scale,
            env=env,
        )
        visualize_samples('Inputs_'+str(tracker.epoch), s_inputs, env, vis)
        visualize_samples('Target_'+str(tracker.epoch), t_inputs, env, vis)

    else:
        scale_plot=vis.boxplot(
            X=s_scale,
            env=env,
            win=scale_plot,
        )

    visualize_samples('Transported_'+str(tracker.epoch), s_generated, env, vis)

    return W1Plot, scale_plot

def visualize_samples(title, sample, env, vis):
    vis.images(
        sample,
        opts=dict(title=title),
        env=env
        )
    vis.save([env])
