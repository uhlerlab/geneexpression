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


# ============ PARSE ARGUMENTS =============

args = utils.setup_args()
args.save_name = args.save_file + args.env
print(args)

# ============ GRADIENT PENALTY (for discriminator) ================


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * args.lambG

    return gradient_penalty


def calc_gradient_penalty_rho(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    _, disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if torch.cuda.is_available() else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * args.lambG2
    return gradient_penalty


# ============= TRAINING INITIALIZATION ==============
# initialize discriminator
netD = GAN.Discriminator(args.nz, args.n_hidden)
print("Discriminator loaded")

# initialize generator
netG = GAN.Generator(args.nz, args.n_hidden)
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    print("Using GPU")

# load data

loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')
sys.stdout.flush()
# setup optimizers
G_opt = optim.Adam(list(netG.parameters()), lr=args.lrG)
D_opt = optim.Adam(list(netD.parameters()), lr=args.lrD)

# loss criteria
logsigmoid = nn.LogSigmoid()
mse = nn.MSELoss()
LOG2 = Variable(torch.from_numpy(np.ones(1)*np.log(2)).float())
print(LOG2)
if torch.cuda.is_available():
    LOG2 = LOG2.cuda()

# =========== LOGGING INITIALIZATION ================

vis = utils.init_visdom(args.env)
tracker = utils.Tracker()
tracker_plot = None
scale_plot = None

# ============================================================
# ============ MAIN TRAINING LOOP ============================
# ============================================================

for epoch in range(args.max_iter):
  
    for it, (s_inputs, t_inputs) in enumerate(loader):
        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()

        if it % args.critic_iter == args.critic_iter-1:
            netG.train()
            netG.zero_grad()

            s_outputs, s_scale  = netG(s_inputs)
           
            G_loss = mse(s_outputs, t_inputs)
            G_loss.backward()
            G_opt.step()
    netG.eval()
    for s_inputs, t_inputs in loader:
        num = s_inputs.size(0)
        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
                s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()
        s_outputs, s_scale = netG(s_inputs)
        W_loss = mse(s_inputs, s_outputs)
        tracker.add(W_loss.cpu().data, num)
        # save transported points
        if epoch % 10 == 0 and epoch > 200:
            with open(args.save_name+str(epoch)+"_trans.txt", 'ab') as f:
                np.savetxt(f, s_outputs.cpu().data.numpy(), fmt='%f')

    tracker.tick()

       
    torch.save(netG.cpu().state_dict(), args.save_name+"_netG.pth")

    if torch.cuda.is_available():
        netG.cuda()
       
    with open(args.save_name+"_tracker.pkl", 'wb') as f:
        pickle.dump(tracker, f)
    if epoch % 5 == 0 and epoch > 5:
        utils.plot(tracker, epoch, t_inputs.cpu().data.numpy(), args.env, vis)



