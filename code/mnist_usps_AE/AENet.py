import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, nc=1, dim=16, latent_size=32):
        super(VAE, self).__init__()
        
        self.nc = nc
        self.dim = dim
        self.input_size = dim*dim
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
        )

        self.mu = nn.Linear(self.input_size, self.latent_size)
        self.logvar = nn.Linear(self.input_size, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.input_size),
            nn.Linear(self.input_size, self.input_size)
        )

    def encode(self, x):
        x = x.view(-1, self.input_size)
        return self.mu(x), self.logvar(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.decoder(z)
        return z.view(-1, self.dim, self.dim)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def generate(self, z):
        z = self.decode(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
