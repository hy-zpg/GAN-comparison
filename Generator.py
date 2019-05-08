import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn


##### DCGAN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
# G(z)
class DCGANgenerator(nn.Module):
    # initializers
    def __init__(self, d=128,channel=1):
        super(DCGANgenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, channel, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x


### channel adjustable DCGAN, multi-gpu
class adDCGANGenerator(nn.Module):
    def __init__(self, ngpu,nz,nc,ngf):
        super(adDCGANGenerator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output





###### another DCGAN
class DCGANGenerator(nn.Module):
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(DCGANGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.height = height
        self.length = length
        self.mult = 2**blocks

        self.initial_linear = nn.Linear(input_size, hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_activ = nn.PReLU(hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_norm = nn.LayerNorm(hidden_size * self.mult * height//self.mult * length//self.mult)

        self.convs = nn.ModuleList([nn.Conv2d(hidden_size * 2 **(blocks - i), hidden_size * 2**(blocks - i - 1), (5, 5), padding=(2, 2)) for i in range(blocks)])
        self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2**(blocks - i - 1)) for i in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i - 1), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                       range(blocks)])

        self.final_conv = nn.Conv2d(hidden_size, image_channels, (5, 5), padding=(2, 2))
        self.final_activ = nn.Tanh()

    def forward(self, inputs):
        x = self.initial_linear(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)
        x = x.view(x.shape[0], self.hidden_size * self.mult, self.height//self.mult, self.length//self.mult)

        for i in range(self.blocks):
            x = self.convs[i](x)
            x = self.activ[i](x)
            x = self.norm[i](x)
            x = F.upsample(x, scale_factor=2)
        x = self.final_conv(x)
        x = self.final_activ(x)
        return x




##### WGAN-GP
class ResNetGenerator(nn.Module):
    def __init__(self, input_size,  image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(ResNetGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks
        self.height = height
        self.length = length
        self.mult = 2**blocks

        self.initial_linear = nn.Linear(input_size, hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_norm = nn.LayerNorm(hidden_size * self.mult * height//self.mult * length//self.mult)
        self.initial_activ = nn.PReLU(hidden_size * self.mult * height//self.mult * length//self.mult)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ1 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ2 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.convs3 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ3 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.convs4 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i), (3, 3), padding=(1, 1)) for i
             in range(blocks)])
        self.norm4 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.activ4 = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i)) for i in range(blocks)])

        self.transitions_conv = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** (blocks - i), hidden_size * 2 ** (blocks - i - 1), (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.transitions_norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (blocks - i - 1), height // (2 ** (blocks - i)), length // (2 ** (blocks - i))]) for i in
                                    range(blocks)])
        self.transitions_activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (blocks - i - 1)) for i in range(blocks)])

        self.final_conv = nn.Conv2d(hidden_size, image_channels, (5, 5), padding=(2, 2))
        self.final_activ = nn.Tanh()

    def forward(self, inputs):
        x = self.initial_linear(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)

        x = x.view(x.shape[0], self.hidden_size * self.mult, self.height//self.mult, self.length//self.mult)

        for i in range(self.blocks):
            fx = self.convs1[i](x)
            fx = self.activ1[i](fx)
            fx = self.norm1[i](fx)
            fx = self.convs2[i](fx)
            fx = self.activ2[i](fx)
            fx = self.norm2[i](fx)

            x = x + fx

            fx = self.convs3[i](x)
            fx = self.activ3[i](fx)
            fx = self.norm3[i](fx)
            fx = self.convs4[i](fx)
            fx = self.activ4[i](fx)
            fx = self.norm4[i](fx)

            x = x + fx

            x = self.transitions_conv[i](x)
            x = self.transitions_activ[i](x)
            x = self.transitions_norm[i](x)
            x = F.upsample(x, scale_factor=2)

        x = self.final_conv(x)
        x = self.final_activ(x)

        return x
