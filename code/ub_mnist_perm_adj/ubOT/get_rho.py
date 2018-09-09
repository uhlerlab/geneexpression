import torch
from torch import nn, optim
from torch.autograd import Variable, grad

import GAN
import utils_rho as utils
import visdom

import numpy as np
import sys
import os
import pickle

torch.manual_seed(1)

#============ PARSE ARGUMENTS =============

args = utils.setup_args()
args.save_name = args.save_file + args.env
print(args)

#============= TRAINING INITIALIZATION ==============

# initialize discriminator
netD = GAN.Discriminator(args.nz, args.n_hidden)
netD.load_state_dict(torch.load(args.save_name+"_netD.pth"))
print("Discriminator loaded")

# initialize generator
netG = GAN.Generator(args.nz, args.n_hidden)
netG.load_state_dict(torch.load(args.save_name+"_netG.pth"))
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()

# load data
loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')
sys.stdout.flush()

netD.eval()
netG.eval()

for s_inputs, labels in loader:
    num = s_inputs.size(0)
    s_inputs, labels = Variable(s_inputs), Variable(labels)
    if torch.cuda.is_available():
        s_inputs, labels = s_inputs.cuda(), labels.cuda()
    _, s_scale = netG(s_inputs)

    with open(args.save_name+"_rho.txt", 'ab') as f:
        np.savetxt(f, s_scale.cpu().data.numpy(), fmt='%f')
    with open(args.save_name+"_labels.txt", 'ab') as f:
        np.savetxt(f, labels.cpu().data.numpy(), fmt='%f')
