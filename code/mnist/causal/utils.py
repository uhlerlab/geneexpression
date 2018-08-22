import numpy as np
import visdom
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#============= ARGUMENT PARSING ==============

def setup_args():

    options = argparse.ArgumentParser()

    #setup
    options.add_argument('-sd', action="store", dest="save_file", default="./results/")
    options.add_argument('-env', action="store", dest="env", default="ubOT_mnist")

    # training arguments
    options.add_argument('-bs', action="store", dest="batch_size", default = 64, type = int)
    options.add_argument('-iter', action="store", dest="max_iter", default = 100, type = int)
    options.add_argument('-citer', action="store", dest="critic_iter", default = 5, type = int)
    options.add_argument('-psi', action="store", dest="psi2", default="JS")
    options.add_argument('-lrG', action="store", dest="lrG", default=1e-4, type = float)
    options.add_argument('-lrD', action="store", dest="lrD", default=1e-4, type = float)
    
    return options.parse_args()

#============= DATA LOADING ==============

class MNISTDataset(Dataset):

    def __init__(self, data1, data2, mode = "train"):
        self.data1 = data1
        self.data2 = data2
        self.transform = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
        self.mode = mode

    def __len__(self):
        return self.data1.size(0)

    def __getitem__(self, idx):
        
        img1 = self.data1[idx]
        img1 = Image.fromarray(img1.numpy(), mode='L')
        if self.mode == "train":
            img2 = self.data2[idx]
            img2 = Image.fromarray(img2.numpy(), mode='L')
            return (self.transform(img1), self.transform(img2))
        else:
            lab1 = self.data2[idx]
            return (self.transform(img1), lab1)
        
def setup_data_loaders(batch_size):
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    train_data = trainset.train_data
    train_label = trainset.train_labels

    # separate into classes
    traindata_classes = []
    trainlabels_classes = []
    for idx in range(10):
        traindata_classes.append(train_data[train_label==idx])
        trainlabels_classes.append(train_label[train_label==idx])
        print(train_data[train_label==idx].shape)

    # separate into two datasets
    splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]
    traindata_1 = []
    traindata_2 = []
    trainlabels_1 = []
    trainlabels_2 = []

    for idx in range(10):
        s = splits[idx]
        data = traindata_classes[idx]
        labels = trainlabels_classes[idx]
        n = 5421
        data = data[0:n]
        labels = labels[0:n]
        traindata_1.append(data[0:int(s*n)])
        print(data[0:int(s*n)].size())
        trainlabels_1.append(labels[0:int(s*n)])
        traindata_2.append(data[int(s*n):])
        print(data[int(s*n):].size())
        trainlabels_2.append(labels[int(s*n):])
        
    traindata_1 = torch.cat(traindata_1)
    trainlabels_1 = torch.cat(trainlabels_1)
    traindata_2 = torch.cat(traindata_2)
    trainlabels_2 = torch.cat(trainlabels_2)

    #scramble
    perm1 = torch.randperm(traindata_1.size(0))
    perm2 = torch.randperm(traindata_2.size(0))
    traindata_1 = traindata_1[perm1]
    trainlabels_1 = trainlabels_1[perm1]
    traindata_2 = traindata_2[perm2]
    trainlabels_2 = trainlabels_2[perm2]

    traindata_2 = traindata_2[0:traindata_1.size(0)]
    trainlabels_2 = trainlabels_2[0:traindata_1.size(0)]

    train_dataset = MNISTDataset(traindata_1, traindata_2)
    eval_dataset = MNISTDataset(traindata_1, trainlabels_1, mode="test")

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return trainloader, evalloader

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


def plot(tracker, env, vis):
    # close old plots
    vis.close(env=env)
    # update plots
    W1Plot=vis.line(
        Y=np.array(tracker.get_status()),
        X=np.array(range(tracker.epoch)),
        env=env,
        name="Loss",

    )
