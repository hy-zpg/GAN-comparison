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

from Discriminator import DCGANdiscriminator
from Generator import DCGANgenerator
from visualization import DCGAN_show_result,show_train_hist
from data_utils import DCGAN_data_preparation
import metric


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 20
img_size = 64
isCrop = False
z_shape = 100

# data_loader
# data_dir = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/FIGR/data/FIGR-8/Data'       # this path depends on your computer
data_dir = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/FIGR/data/omniglot-py'
# data_dir = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/Matching-network-GAN/datasets/FIGR-8'
train_loader = DCGAN_data_preparation(data_dir,img_size,batch_size)




G = DCGANgenerator(128)
D = DCGANdiscriminator(128)
# print(G)
# print(D)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G=nn.DataParallel(G,device_ids=[0,1])
D=nn.DataParallel(D,device_ids=[0,1])

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer for two networks
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
score_tr = np.zeros((train_epoch, 4*7+3))
s = metric.compute_score_raw('figr', img_size, data_dir, 2000, 16, 'Runs/DCGAN_FIGR_Results/real/','Runs/WGAN_FIGR_Results/fake/',
                             G,z_shape, conv_model='inception_v3', workers=4)
score_tr[0] = s
np.save('%s/score_tr.npy' % ('Runs/DCGAN_FIGR_Results'), score_tr)


train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('Training start!')
start_time = time.time()
for epoch in range(train_epoch):
    print('this is epoch {}'.format(epoch))
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    ### DCGAN
    ### Discriminator: can make accurate distinguish, D_train_loss = fake_loss(D(G(z_)),y_fake_) + real_loss(D(x_),y_real_), training discriminator first
    ### Generator: can generate the same fake image as real image, G_train_loss = (D(G(z_),y_real_), then training generator
    print('starting training')
    num_iter = 0
    epoch_start_time = time.time()
    print(len(train_loader))
    for x_, _ in train_loader:
        # train discriminator D
        D.zero_grad()
        
        if isCrop:
            x_ = x_[:, :, 22:86, 22:86]

        mini_batch = x_.size()[0]

        y_real_ = torch.ones(mini_batch)
        y_fake_ = torch.zeros(mini_batch)

        x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
        # [128, 3, 64, 64]
        D_result = D(x_).squeeze()
        # [128]
        D_real_loss = BCE_loss(D_result, y_real_)

        z_ = torch.randn((mini_batch, z_shape)).view(-1, z_shape, 1, 1)
        z_ = Variable(z_.cuda())
        G_result = G(z_)
        # [128, 3, 64, 64]

        D_result = D(G_result).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake_)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        D_losses.append(D_train_loss.data)

        # train generator G
        G.zero_grad()

        z_ = torch.randn((mini_batch, z_shape)).view(-1, z_shape, 1, 1)
        z_ = Variable(z_.cuda())

        G_result = G(z_)
        D_result = D(G_result).squeeze()
        G_train_loss = BCE_loss(D_result, y_real_)
        G_train_loss.backward()
        G_optimizer.step()

        G_losses.append(G_train_loss.data)

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    if not os.path.isdir('Runs/DCGAN_FIGR_Results'):
        os.mkdir('Runs/DCGAN_FIGR_Results')
    p = 'Runs/DCGAN_FIGR_Results/' + str(epoch + 1) + '.png'
    DCGAN_show_result((epoch+1),G, p, 5, z_shape)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)


    ## evaluation
    s = metric.compute_score_raw('figr', img_size, data_dir, 200, 16, 'Runs/DCGAN_FIGR_Results/real/','Runs/WGAN_FIGR_Results/fake/',
                                 G,z_shape, conv_model='inception_v3', workers=4)
    score_tr[epoch] = s

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "Runs/DCGAN_FIGR_Results/generator_param.pkl")
torch.save(D.state_dict(), "Runs/DCGAN_FIGR_Results/discriminator_param.pkl")
with open('Runs/DCGAN_FIGR_Results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='Runs/DCGAN_FIGR_Results/train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'Runs/DCGAN_FIGR_Results/' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('Runs/DCGAN_FIGR_Results/generation_animation.gif', images, fps=5)
