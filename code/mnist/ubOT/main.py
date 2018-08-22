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


#============= TRAINING INITIALIZATION ==============


# initialize discriminator
netD = GAN.Discriminator()
print("Discriminator loaded")

# initialize generator
netG = GAN.Generator()
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()

# load data
loader, evalloader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')
sys.stdout.flush()

# setup optimizers
G_opt = optim.Adam(list(netG.parameters()), lr = args.lrG)
D_opt = optim.Adam(list(netD.parameters()), lr = args.lrD)

# loss criteria
logsigmoid = nn.LogSigmoid()
#mse = nn.MSELoss(reduce=False)
LOG2 = Variable(torch.from_numpy(np.ones(1)*np.log(2)).float())
if torch.cuda.is_available():
    LOG2 = LOG2.cuda()

def mse(tensor1, tensor2):
    mse_loss = nn.MSELoss(reduce=False)
    error = mse_loss(tensor1.view(-1,784), tensor2.view(-1,784))
    return error

#=========== LOGGING INITIALIZATION ================

vis = utils.init_visdom(args.env)
tracker = utils.Tracker()

#============================================================
#============ MAIN TRAINING LOOP ============================
#============================================================

for epoch in range(args.max_iter):

    for it, (s_inputs, t_inputs) in enumerate(loader):

        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()

#================== Train generator =========================
        if it % args.critic_iter == args.critic_iter-1:
            netG.train()
            netD.eval()

            netG.zero_grad()

            # pass source inputs through generator network
            s_generated, s_scale = netG(s_inputs)

            # pass generated source data and target inputs through discriminator network
            s_outputs = netD(s_generated)

            # compute loss
            G_loss = args.lamb0*torch.mean(torch.sum(mse(s_generated, s_inputs), dim=1)) #+ args.lamb1*torch.mean(-torch.log(s_scale)+s_scale)

#            if args.psi2 == "EQ":
#                G_loss += - args.lamb2*torch.mean(s_scale*s_outputs)
#            else:
#                G_loss += args.lamb2*torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs))

            # update params
            G_loss.backward()
            G_opt.step()


#================== Train discriminator =========================
        else:
            netD.train()
            netG.eval()

            netD.zero_grad()

            # pass source inputs through generator network
            s_generated, s_scale = netG(s_inputs)

            # pass generated source data and target inputs through discriminator network
            s_outputs, t_outputs = netD(s_generated), netD(t_inputs)

            # compute loss
            D_loss = 0
            if args.psi2 == "EQ":
                D_loss += torch.mean(s_scale*s_outputs) - torch.mean(t_outputs)
            else:
                D_loss += -torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs)) - torch.mean(LOG2.expand_as(t_outputs)+logsigmoid(t_outputs)) #+ calc_gradient_penalty(netD, s_generated.data, t_inputs.data)

            # update params
            D_loss.backward()
            D_opt.step()

#================= Log results ===========================================

    netD.eval()
    netG.eval()

    for s_inputs, t_inputs in loader:
        num = s_inputs.size(0)
        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()

        s_generated, s_scale = netG(s_inputs)
        s_outputs, t_outputs = netD(s_generated), netD(t_inputs)

        # update tracker
        W_loss = args.lamb0*torch.mean(torch.sum(mse(s_generated, s_inputs), dim=1)) #+ args.lamb1*torch.mean(-torch.log(s_scale)+s_scale)
#        W_loss += torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs))
 #       W_loss += torch.mean(LOG2.expand_as(t_outputs)+logsigmoid(t_outputs))
        tracker.add(W_loss.cpu().data, num)

    tracker.tick()

    for s_inputs, s_labels in evalloader:
        num = s_inputs.size(0)
        s_inputs = Variable(s_inputs)
        if torch.cuda.is_available():
            s_inputs = s_inputs.cuda()

        _, s_scale = netG(s_inputs)

        # save transported points
        if epoch%1==0:
            with open(args.save_name+str(epoch)+"_rho.txt", 'ab') as f:
                np.savetxt(f, s_scale.cpu().data.numpy(), fmt='%f')
            with open(args.save_name+str(epoch)+"_labels.txt", 'ab') as f:
                np.savetxt(f, s_labels.cpu().data.numpy(), fmt='%f')


    # save models
    torch.save(netD.cpu().state_dict(), args.save_name+"_netD.pth")
    torch.save(netD.cpu().state_dict(), args.save_name+"_netG.pth")

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()

    # save tracker
    with open(args.save_name+"_tracker.pkl", 'wb') as f:
        pickle.dump(tracker, f)

    # visualize progress
    if epoch % 1 == 0:
        utils.plot(tracker, s_inputs.cpu().data, s_generated.cpu().data, args.env, vis)