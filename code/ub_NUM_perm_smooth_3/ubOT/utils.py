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
#    options.add_argument('-dat1', action="store", dest="source_data_file", default="../../../datasets/mnist_usps/latents_data/ub_mnist_data.npy")
#    options.add_argument('-dat2', action="store", dest="target_data_file", default="../../../datasets/mnist_usps/latents_data/usps_data.npy")
    options.add_argument('-env', action="store", dest="env", default="OT_unbalanced_NUM_perm")
    options.add_argument('-bri', action="store", dest="bri", default=1., type = float)

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

class CombinedDataset(Dataset):

    def __init__(self):
        self.data1, self.labels1 = self.get_MNIST()
        self.data2, self.labels2 = self.get_USPS()
        self.length = min(len(self.data1), len(self.data2))
        self.data1, self.labels1 = self.data1[0:self.length], self.labels1[0:self.length]
        self.data2, self.labels2 = self.data2[0:self.length], self.labels2[0:self.length]

        self.transform = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])

    def get_USPS(self):
        file_path = './data/usps_resampled.mat'
        data = loadmat(file_path)
        raw_images = np.concatenate((data['train_patterns'].T, data['test_patterns'].T))
        raw_labels = np.concatenate((data['train_labels'].T, data['test_labels'].T))

        data_length = raw_images.shape[0]
        data_dim = int(np.sqrt(raw_images.shape[1]))

        images = []
        labels = []
        for i in range(data_length):
            images.append(np.uint8(((raw_images[i].reshape(data_dim, data_dim)+1)/2)*255))
            labels.append(np.where(raw_labels[i]==1)[0][0])

        images = np.array(images)
        labels = np.array(labels)

        rand = random.permutation(len(images))
        images = images[rand]
        labels = labels[rand]
        return images, labels

    def get_MNIST(self):
        # extract mnist datasets
        mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        mnist_data = list(mnist_trainset.train_data.numpy())
        mnist_labels = list(mnist_trainset.train_labels.numpy())

        mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        mnist_data += list(mnist_testset.test_data.numpy())
        mnist_labels += list(mnist_testset.test_labels.numpy())

        mnist_data = np.array(mnist_data)
        mnist_labels = np.array(mnist_labels)

        rand = random.permutation(len(mnist_data))
        mnist_data = mnist_data[rand]
        mnist_labels = mnist_labels[rand]

        return mnist_data, mnist_labels

    def get_stats(self, labels):
        print(Counter(labels))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img1, img2 = self.data1[idx], self.data2[idx]
        img1, img2 = Image.fromarray(img1, mode='L'), Image.fromarray(img2, mode='L')
        img1, img2  = self.transform(img1), self.transform(img2)
        return (img1.view(-1), img2.view(-1))


def setup_data_loaders(batch_size):
    trainset = CombinedDataset()
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader



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
    else:
        scale_plot=vis.boxplot(
            X=s_scale,
            env=env,
            win=scale_plot,
        )


    visualize_samples('Inputs_'+str(tracker.epoch), s_inputs, env, vis)
    visualize_samples('Transported_'+str(tracker.epoch), s_generated, env, vis)
    visualize_samples('Target_'+str(tracker.epoch), t_inputs, env, vis)
    return W1Plot, scale_plot

def visualize_samples(title, sample, env, vis):
    vis.images(
        sample,
        opts=dict(title=title),
        env=env
        )
    vis.save([env])
