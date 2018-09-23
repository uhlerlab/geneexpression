import torch
from torch import nn, optim
from torch.autograd import Variable, grad

import GAN
import utils
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

# initialize discriminator
netD = GAN.Discriminator(args.nz, args.n_hidden)
netD.load_state_dict(torch.load(args.save_name+"_netD.pth"))
print("Discriminator loaded")

# initialize generator
netG = GAN.Generator(args.nz, args.n_hidden)
netG.load_state_dict(torch.load(args.save_name+"_netG.pth"))
netGS = GAN.Generator_Scale(args.nz, args.n_hidden)
netGS.load_state_dict(torch.load(args.save_name+"_netGS.pth"))
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    netGS.cuda()

netD.eval()
netG.eval()
netGS.eval()

# load data
loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')
sys.stdout.flush()

#================= Log results ===========================================

for s_inputs, t_inputs in loader:
    s_inputs = Variable(s_inputs)
    if torch.cuda.is_available():
        s_inputs = s_inputs.cuda()
    s_generated, s_scale = netG(s_inputs), netGS(s_inputs)

    with open(args.save_name+"_rho.txt", 'ab') as f:
        np.savetxt(f, s_scale.cpu().data.numpy(), fmt='%f')
    with open(args.save_name+"_trans.txt", 'ab') as f:
        np.savetxt(f, s_generated.cpu().data.numpy(), fmt='%f')
    with open(args.save_name+"_input.txt", 'ab') as f:
        np.savetxt(f, s_inputs.cpu().data.numpy(), fmt='%f')
    with open(args.save_name+"_target.txt", 'ab') as f:
        np.savetxt(f, t_inputs.numpy(), fmt='%f')

