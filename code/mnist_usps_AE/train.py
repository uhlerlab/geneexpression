import torch
from torch import nn, optim
from torch.autograd import Variable
import AENet
import argparse
import utils as utils
import visualize
import numpy as np
import sys

# parse arguments
def setup_args():

    options = argparse.ArgumentParser()

    options.add_argument('-sf', action="store", dest="save_file", default = "num_lamb_0.001")
    options.add_argument('-pt', action="store", dest="pretrained_file", default=None)
    options.add_argument('-bs', action="store", dest="batch_size", default = 128, type = int)
    options.add_argument('-env', action="store", dest="env", default="VAE_MNIST_USPS")

    options.add_argument('-iter', action="store", dest="max_iter", default = 200, type = int)
    options.add_argument('-lr', action="store", dest="lr", default=1e-3, type = float)
    options.add_argument('-nz', action="store", dest="nz", default=20, type = int)
    options.add_argument('-lamb', action="store", dest="lamb", default=0.001, type = float)

    return options.parse_args()

args = setup_args()
print(args)
sys.stdout.flush()

# retrieve dataloaders
train_loader, test_loader = utils.setup_data_loaders(args.batch_size)
print('Data loaded')

model = AENet.VAE(nc=1, latent_size=args.nz)
if args.pretrained_file is not None:
    model.load_state_dict(torch.load(args.pretrained_file))
    print("Pre-trained model loaded")
    sys.stdout.flush()

if torch.cuda.is_available():
    print('Using GPU')
    model.cuda()

optimizer = optim.Adam([
    {'params': model.parameters()}],
    lr = args.lr)

def loss_function(recon_x, x, mu, logvar, latents):
    MSE = nn.MSELoss()
    lloss = MSE(recon_x,x)
    if args.lamb>0:
        KL_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lloss = lloss + args.lamb*KL_loss
    return lloss

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, _) in enumerate(train_loader):

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        optimizer.zero_grad()
        recon_inputs, latents, mu, logvar = model(inputs)

        loss = loss_function(recon_inputs, inputs, mu, logvar, latents)
        train_loss += loss.data[0]

        loss.backward()
        optimizer.step()

    print('Epoch: {} Average loss: {:.15f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (inputs, _) in enumerate(test_loader):

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        recon_inputs, latents, mu, logvar = model(inputs)

        loss = loss_function(recon_inputs, inputs, mu, logvar, latents)
        test_loss += loss.data[0]

    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.15f}'.format(test_loss))
    return test_loss


def save(epoch):
    torch.save(model.cpu().state_dict(), args.save_file+'_'+str(epoch)+".pth")
    if torch.cuda.is_available():
        model.cuda()

def generate_image(epoch):

    num_imgs = 1

    for i in range(5):
        inputs, _ = train_loader.dataset[np.random.randint(100)]
        inputs = Variable(inputs.view(1,1,16,16))

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        recon_inputs, _, _, _ = model(inputs)

        visualize.visualize_samples("Train "+str(epoch)+" inputs", inputs.cpu().data.view(num_imgs,1,16,16), args.env)
        visualize.visualize_samples("Train "+str(epoch)+" recon", recon_inputs.cpu().data.view(num_imgs,1,16,16), args.env)

        inputs, _ = test_loader.dataset[np.random.randint(100)]
        inputs = Variable(inputs.view(1,1,16,16))

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        recon_inputs, _, _, _ = model(inputs)

        visualize.visualize_samples("Test "+str(epoch)+" inputs", inputs.cpu().data.view(num_imgs, 1, 64, 64), args.env)
        visualize.visualize_samples("Test "+str(epoch)+" recon", recon_inputs.cpu().data.view(num_imgs, 1, 64, 64), args.env)

# main training loop
visualize.reset(args.env)
generate_image(0)
save(0)

for epoch in range(args.max_iter):
    train(epoch)
    _ = test(epoch)

    if epoch % 10 == 0:
        generate_image(epoch)
        save(epoch)
