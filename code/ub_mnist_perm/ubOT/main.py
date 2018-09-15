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

#============ GRADIENT PENALTY (for discriminator) ================

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

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * args.lambG
    return gradient_penalty


#============= TRAINING INITIALIZATION ==============

# initialize discriminator
netD = GAN.Discriminator(args.nz, args.n_hidden)
print("Discriminator loaded")

# initialize generator
netG = GAN.Generator(args.nz, args.n_hidden)
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()

# load data
loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')
sys.stdout.flush()

# setup optimizers
G_opt = optim.Adam(list(netG.parameters()), lr = args.lrG)
D_opt = optim.Adam(list(netD.parameters()), lr = args.lrD)

# loss criteria
logsigmoid = nn.LogSigmoid()
mse = nn.MSELoss(reduce=False)
LOG2 = Variable(torch.from_numpy(np.ones(1)*np.log(2)).float())
print(LOG2)
if torch.cuda.is_available():
    LOG2 = LOG2.cuda()

#=========== LOGGING INITIALIZATION ================

vis = utils.init_visdom(args.env)
tracker = utils.Tracker()
tracker_plot=None
scale_plot=None

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
#            G_loss = torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) + args.lamb1*torch.mean(-torch.log(s_scale)+s_scale)
#            G_loss = torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) - args.lamb1*torch.mean(s_scale*(1-torch.log(s_scale)))
            G_loss = args.lamb0*torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) + args.lamb1*torch.mean((s_scale-1)*torch.log(s_scale))

            if args.psi2 == "EQ":
                G_loss += - args.lamb2*torch.mean(s_scale*s_outputs)
            else:
                G_loss += args.lamb2*torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs))

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
            #D_loss = 0
            D_loss = calc_gradient_penalty(netD, s_generated.data, t_inputs.data)
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
        W_loss = args.lamb0*torch.mean(s_scale*torch.sum(mse(s_generated, s_inputs), dim=1)) + args.lamb1*torch.mean(-torch.log(s_scale)+s_scale)
        W_loss += torch.mean(s_scale*(LOG2.expand_as(s_outputs)+logsigmoid(s_outputs)-s_outputs))
        W_loss += torch.mean(LOG2.expand_as(t_outputs)+logsigmoid(t_outputs))
        tracker.add(W_loss.cpu().data, num)

    tracker.tick()

    # save models
    torch.save(netD.cpu().state_dict(), args.save_name+"_netD.pth")
    torch.save(netG.cpu().state_dict(), args.save_name+"_netG.pth")

    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()

    # save tracker
    with open(args.save_name+"_tracker.pkl", 'wb') as f:
        pickle.dump(tracker, f)

    # visualize progress
    im_inputs = s_inputs.view(-1, 1, 16, 16)
    im_trans = s_generated.view(-1, 1, 16, 16)
    im_targets = t_inputs.view(-1, 1, 16, 16)

    if epoch % 50 == 0:
        tracker_plot, scale_plot = utils.plot(tracker, tracker_plot, scale_plot, im_inputs.cpu().data.numpy(), im_trans.cpu().data.numpy(), s_scale.cpu().data.numpy(), im_targets.cpu().data.numpy(), args.env, vis)