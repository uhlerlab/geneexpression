import torch
import torch.nn as nn
from torch.autograd import Variable

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size, batchnorm=False):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm

        self.encoder = nn.Sequential(
            # input is 3 x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (nc) x 128 x 128
        )

        self.d1 = nn.Sequential(
            nn.Linear(latent_variable_size, ngf*8*4*4),
            nn.ReLU(inplace=True),
            )
        self.bn_mean = nn.BatchNorm1d(latent_variable_size)

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.ndf*8*4*4)
        if self.batchnorm:
            return self.bn_mean(self.fc1(h)), self.fc2(h)
        else:
            return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = self.d1(z)
        h = h.view(-1, self.ngf*8, 4, 4)
        return self.decoder(h)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, 128, 128))
        z = self.reparametrize(mu, logvar)
        return z

    def generate(self, z):
        res = self.decode(z)
        return res

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, 128, 128))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
