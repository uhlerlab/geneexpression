import torch
from torch import nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, nz=256, dim=256):
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
    def __init__(self, input_size=256, nz=64):
        super(Generator, self).__init__()

        self.nz = nz
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.nz),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.nz, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Generator_Scale(nn.Module):
    def __init__(self, input_size=256, nz=64):
        super(Generator_Scale, self).__init__()

        self.nz = nz
        self.input_size = input_size

        self.scale = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, 1),
            nn.Softplus(),
            )

    def forward(self, x):
        return self.scale(x)

