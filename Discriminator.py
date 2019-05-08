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

#####DCGAN
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
class DCGANdiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128,channel=1):
        super(DCGANdiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(channel, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x


## channel adjustable DCGAN, multi-gpu
class adDCGANDiscriminator(nn.Module):
    def __init__(self, ngpu,nz,nc,ndf):
        super(adDCGANDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)



######another DCGAN
class DCGANDiscriminator(nn.Module):
    def __init__(self, image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(DCGANDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks



        self.initial_conv = nn.Conv2d(image_channels, hidden_size, (5, 5), padding=(2, 2))
        self.initial_norm = nn.LayerNorm([hidden_size, height, length])
        self.initial_activ = nn.PReLU(hidden_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i + 1), (5, 5), padding=(2, 2)) for
             i in range(blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (i + 1), height // (2 ** i), length // (2 ** i)]) for i
                                   in range(blocks)])
        self.activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])

        self.final_linear = nn.Linear(hidden_size * 2 ** blocks * height//(2**blocks) * length//(2**blocks), 1)

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_norm(x)
        x = self.initial_activ(x)

        for i in range(self.blocks):
            x = self.convs[i](x)
            x = self.norm[i](x)
            x = self.activ[i](x)
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x


###### WGAN-GP
class ResNetDiscriminator(nn.Module):
    def __init__(self, image_channels=1, height=32, length=32, hidden_size=64, blocks=4):
        super(ResNetDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks

        self.initial_conv = nn.Conv2d(image_channels, hidden_size, (7, 7), padding=(3, 3))
        self.initial_norm = nn.LayerNorm([hidden_size, height, length])
        self.initial_activ = nn.PReLU(hidden_size)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ1 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ2 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs3 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ3 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs4 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.norm4 = nn.ModuleList([nn.LayerNorm(
            [hidden_size * (2 ** i), height // (2 ** i), length // (2 ** i)]) for i
            in range(blocks)])
        self.activ4 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.transitions_conv = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i+1), (3, 3), padding=(1, 1)) for
             i in range(blocks)])
        self.transitions_norm = nn.ModuleList([nn.LayerNorm(
            [hidden_size * 2 ** (i + 1), height // (2 ** i), length // (2 ** i)]) for i in
                                    range(blocks)])
        self.transitions_activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])
        self.final_linear = nn.Linear(hidden_size * 2 ** blocks, 1)

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_activ(x)
        x = self.initial_norm(x)

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
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)

        return x
