import torch
from torch import nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, nz=2, dim=512):
        super(Discriminator, self).__init__()
        self.nz = nz
        self.hiddenz = dim
        self.net = nn.Sequential(
            nn.Linear(self.nz, self.hiddenz),
            nn.ReLU(),
            nn.Linear(self.hiddenz, self.hiddenz),
            nn.ReLU(),
            nn.Linear(self.hiddenz, self.hiddenz),
            nn.ReLU(),
            nn.Linear(self.hiddenz, 1),
            )

    def forward(self, x):
        x = x.view(-1, self.nz)
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, nz=2, dim=512):
        super(Generator, self).__init__()
        self.nz = nz
        self.hiddenz = dim
        self.net = nn.Sequential(
            nn.Linear(self.nz, self.hiddenz),
            nn.ReLU(),
            nn.Linear(self.hiddenz, self.hiddenz),
            nn.ReLU(),
            nn.Linear(self.hiddenz, self.hiddenz),
            nn.ReLU(),
            )
        self.output = nn.Linear(self.hiddenz, self.nz)
        self.scale = nn.Sequential(
            nn.Linear(self.hiddenz, 1),
            nn.Softplus(),
            )

    def forward(self, z):
        z = z.view(-1, self.nz)
        z = self.net(z)
        return self.output(z), self.scale(z)
