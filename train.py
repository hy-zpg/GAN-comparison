from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.datasets as datasets
import numpy as np

import metric
from metric import make_dataset
import numpy as np
from Discriminator import adDCGANDiscriminator
from Generator import adDCGANGenerator
import metric
from data_utils import one_channel_preparation, three_channel_preparation
from GAN_environments import adDCGAN, simpleDCGAN, WGAN_GP

##problem channel=3, cann't


###script
# python train.py --network='WGAN_GP_DCGAN' --dataset='FIGR' --niter=1 --ndc=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True, help='adDCGAN | WGAN_GP_DCGAN | simpleDCGAN | WGAN_GP_ResNet ')
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake | FIGR | Ominiglot')
    # parser.add_argument('--dataroot', default = '', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--ndc', type=int, default=3)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='RUNs/', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    ########################################################
    #### For evaluation ####
    parser.add_argument('--sampleSize', type=int, default=2000, help='number of samples for evaluation')
    ########################################################

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

   



    if opt.dataset == 'FIGR':
        dataroot = './data/FIGR-8/Data'       
        # dataroot = './data/small-FIGR-8'
    elif opt.dataset == 'Ominiglot':
        dataroot = './data/omniglot-py/images_background'
        

    ## data preparation
    if opt.ndc == 1:
        train_loader = one_channel_preparation(dataroot,opt.imageSize,opt.ndc,opt.batchSize)
    else:
        train_loader = three_channel_preparation(dataroot,opt.imageSize,opt.batchSize)
    

    # for i, data in enumerate(train_loader, 0):
    #     print('data shape',np.shape(data[0]))
        

    outf = opt.outf + opt.dataset + '_'+ str(opt.ndc) +'/' + opt.network
    if not os.path.exists(outf):
        os.makedirs(outf)
    if opt.network == 'adDCGAN':
        adDCGAN(opt,metric,train_loader,dataroot,outf)
    elif opt.network == 'simpleDCGAN':
        simpleDCGAN(opt,metric,train_loader,dataroot,outf)
    elif opt.network == 'WGAN_GP_DCGAN':
        WGAN_GP(opt,metric,train_loader,dataroot,outf,'DCGAN')
    elif opt.network == 'WGAN_GP_ResNet':
        WGAN_GP(opt,metric,train_loader,dataroot,outf,'ResNet')
    else:
        print('adding this network to this project')




