import torch
from torch import nn, optim
from torch.autograd import Variable
import AENet
import argparse
import data_loader_utils as utils
import numpy as np
import sys

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()
    
    options.add_argument('-sf', action="store", dest="save_file", default="./latents/")
    options.add_argument('-pt', action="store", dest="pretrained_file", default="./results/NUM_0.0000001_240.pth")
    options.add_argument('-bs', action="store", dest="batch_size", default = 128, type = int)

    options.add_argument('-nz', action="store", dest="nz", default=30, type = int)
    return options.parse_args()

args = setup_args()
print(args)
sys.stdout.flush()

# retrieve dataloaders
train_loader, test_loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')

# load trained model
model = AENet.VAE(nc=1, dim=16, latent_size=args.nz)
if args.pretrained_file is not None:
    model.load_state_dict(torch.load(args.pretrained_file))
    print("Pre-trained model loaded")
    sys.stdout.flush()

if torch.cuda.is_available():
    print('Using GPU')
    model.cuda()

model.eval()

def save(name, means, logvars, targets):
    with open(args.save_file + '_' + name + '_' + 'means.txt', 'ab') as f:
    	np.savetxt(f, means, fmt='%f')
    with open(args.save_file + '_' + name + '_' + 'logvars.txt', 'ab') as f:
        np.savetxt(f, logvars, fmt='%f')
    with open(args.save_file + '_' + name + '_' + 'targets.txt', 'ab') as f:
    	np.savetxt(f, targets)


def get_latents(name, loader):
    for (inputs, targets) in loader:
        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        _, _, means, logvars = model(inputs)
        save(name, means.cpu().data.numpy(), logvars.cpu().data.numpy(), targets)

get_latents('train', train_loader)
get_latents('test', test_loader)
