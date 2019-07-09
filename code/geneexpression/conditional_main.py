import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import torch.nn.functional as fn

import cGAN
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


def calc_gradient_penalty(netD, real_data, fake_data, clusters):
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size())

    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, clusters)

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
netD = cGAN.Discriminator(5964, 600)
print("Discriminator loaded")

# initialize generator
netG = cGAN.Generator(5964, 500)
print("Generator loaded")

if torch.cuda.is_available():
    netD.cuda()
    netG.cuda()
    print("Using GPU")

# load data

loader = utils.setup_data_loaders_unsupervised(args.batch_size)
print('Data loaded')
sys.stdout.flush()


# setup optimizers
G_opt = optim.Adam(list(netG.parameters()), lr=args.lrG)
D_opt = optim.Adam(list(netD.parameters()), lr=args.lrD)

# loss criteria
logsigmoid = nn.LogSigmoid()
mse = nn.MSELoss(reduce=False)
LOG2 = Variable(torch.from_numpy(np.ones(1)*np.log(2)).float())
print(LOG2)
if torch.cuda.is_available():
    LOG2 = LOG2.cuda()
mb_size = 509
ones_label = Variable(torch.ones(mb_size, 1))
zeros_label = Variable(torch.zeros(mb_size, 1))
#ones_label_2 = Variable(torch.ones(100, 1))

# =========== LOGGING INITIALIZATION ================

vis = utils.init_visdom(args.env)
tracker = utils.Tracker()
tracker_plot = None
scale_plot = None

# ============================================================
# ============ MAIN TRAINING LOOP ============================
# ============================================================

for epoch in range(args.max_iter):
    for it, (s_inputs, clusters, t_inputs) in enumerate(loader):
        #print(epoch, it)
        s_inputs, clusters, t_inputs = Variable(
            s_inputs), Variable(clusters), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, clusters, t_inputs = s_inputs.cuda(), clusters.cuda(), t_inputs.cuda()

    # ================== Train discriminator =========================
        if it % args.critic_iter != args.critic_iter-1:
            netD.train()
            netG.eval()

            netD.zero_grad()

            # pass source inputs through generator network
            s_generated, s_scale = netG(s_inputs, clusters)
            #print(s_generated.shape, t_inputs.shape)

            # pass generated source data and target inputs through discriminator network
            s_outputs, t_outputs = netD(
                s_generated, clusters), netD(t_inputs, clusters)
            D_real = netD(s_inputs, clusters)
            
            D_loss_real = fn.binary_cross_entropy_with_logits(D_real, ones_label)
            
            D_loss_fake = fn.binary_cross_entropy_with_logits(s_outputs, zeros_label)
            D_loss = D_loss_real + D_loss_fake

            D_loss.backward()
            D_opt.step()

    # ================== Train generator =========================

        else:
            netG.train()
            netD.eval()

            netG.zero_grad()

            # pass source inputs through generator network
            s_generated, s_scale = netG(s_inputs, clusters)

            # pass generated source data and target inputs through discriminator network
            s_outputs = netD(s_generated, clusters)
            
            #print(s_outputs.shape, s_generated.shape)
            G_loss = fn.binary_cross_entropy_with_logits(s_outputs, ones_label)

            G_loss.backward()
            G_opt.step()


# ================= Log results ===========================================

    netD.eval()
    netG.eval()

    for s_inputs, clusters, t_inputs in loader:
        num = s_inputs.size(0)
        s_inputs, t_inputs = Variable(s_inputs), Variable(t_inputs)
        if torch.cuda.is_available():
            s_inputs, t_inputs = s_inputs.cuda(), t_inputs.cuda()

        s_generated, s_scale = netG(s_inputs, clusters)
        s_outputs, t_outputs = netD(
            s_generated, clusters), netD(t_inputs, clusters)

        if epoch % 10 == 0 and epoch > 200:
            with open(args.save_name+str(epoch)+"_trans.txt", 'ab') as f:
                np.savetxt(f, s_generated.cpu().data.numpy(), fmt='%f')
        W_loss = torch.mean(LOG2.expand_as(s_outputs) +
                            logsigmoid(s_outputs)-s_outputs)
        W_loss += torch.mean(LOG2.expand_as(t_outputs) +
                             logsigmoid(t_outputs))
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
    if epoch > 250:
        print("epoch is correct")
    if epoch % 5 == 0 and epoch > 5:
        utils.plot(tracker, epoch, t_inputs.cpu().data.numpy(), args.env, vis)
