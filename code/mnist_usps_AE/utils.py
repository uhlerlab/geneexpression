import numpy as np
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.io import loadmat
import numpy.random as random
from collections import Counter

random.seed(0)



#============= DATA LOADING ==============

class CombinedDataset(Dataset):

    def __init__(self, mode = "train", split = 0.9):
        self.data, self.labels = self.combine_data(mode, split)
        self.is_mnist = self.sort_data()
        
        self.transform = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
        self.mode = mode

    def sort_data():
        return [1 if self.data[idx].shape[1]==28 else 0 for idx in range(len(self.data))]

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

    def combine_data(self, mode, split):
        data1, labels1 = self.get_MNIST()
        data2, labels2 = self.get_USPS()
        min_length = min(len(data1), len(data2))

        data1 = data1[0:min_length]
        labels1 = labels1[0:min_length]
        data2 = data2[0:min_length]
        labels2 = labels2[0:min_length]
        
        print("MNIST")
        self.get_stats(labels1)
        print("USPS")
        self.get_stats(labels2)
        
        data = list(data1) + list(data2)
        labels = list(labels1) + list(labels2)

        rand = random.permutation(2*min_length)
        data = [data[i] for i in rand]
        labels = [labels[i] for i in rand]

        split_idx = int(split*2*min_length)
        if mode == "train":
            data = data[0:split_idx]
            labels = labels[0:split_idx]
        else:
            data = data[split_idx:]
            labels = labels[split_idx:]
        
        print("Combined")
        self.get_stats(labels)

        return data, labels

    def get_stats(self, labels):
        print(Counter(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img, mode='L')
        img = self.transform(img)

        return (img, (self.labels[idx], self.is_mnist[idx]))


def setup_data_loaders(batch_size):
    trainset = CombinedDataset()
    testset = CombinedDataset(mode="test")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return trainloader, testloader


def test():
    dataset = CombinedDataset()
    print(np.uint8(dataset.__getitem__(5)[0]*255))


if __name__ == "__main__":
    test()
