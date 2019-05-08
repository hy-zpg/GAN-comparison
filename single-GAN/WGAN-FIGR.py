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

from docopt import docopt
import torch.nn as nn

import torch
import torch.optim as optim
import torch.autograd as autograd

from tensorboardX import SummaryWriter
import numpy as np
import os
from PIL import Image

from Discriminator import ResNetDiscriminator, DCGANDiscriminator
from Generator import ResNetGenerator, DCGANGenerator
from visualization import WGAN_show_result, show_train_hist
from  data_utils import WGAN_data_preparation

import torch.utils.data as data
import os
import os.path
import errno
import metric

### reading images from file


def wassertein_loss(inputs, targets):
    return torch.mean(inputs * targets)


def calc_gradient_penalty(discriminator, real_batch, fake_batch):
    epsilon = torch.rand(real_batch.shape[0], 1, device=device)
    interpolates = epsilon.view(-1, 1, 1, 1) * real_batch + (1 - epsilon).view(-1, 1, 1, 1) * fake_batch
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


def normalize_data(data):
    data *= 2
    data -= 1
    return data


def unnormalize_data(data):
    data += 1
    data /= 2
    return data



#######Starting 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
z_shape = 100
train_epoch = 20

# data_loader


#neural_network = 'DCGAN'
neural_network = 'DCGAN'
learning_rate = 1e-3
channels = 1
height = 32
length = 32



# data_dir = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/FIGR/data/FIGR-8/Data'       # this path depends on your computer
data_dir = '/media/user/05e85ab6-e43e-4f2a-bc7b-fad887cfe312/meta_gan/Matching-network-GAN/datasets/FIGR-8'
train_loader = WGAN_data_preparation(data_dir,length,channels,batch_size)




# prepate network
D = eval(neural_network + 'Discriminator(channels,height, length)').to(device)
G = eval(neural_network + 'Generator(z_shape, channels, height, length)').to(device)
# print(G)
# print(D)
G.cuda()
D.cuda()
G=nn.DataParallel(G,device_ids=[0,1])
D=nn.DataParallel(D,device_ids=[0,1])


## discriminator: [real_image_label(1),fake_image_label(-1)] -> loss
## generator: [real_image_label(1)]
discriminator_targets = torch.tensor([1] * batch_size + [-1] * batch_size, dtype=torch.float, device=device).view(-1, 1)
generator_targets = torch.tensor([1] * batch_size, dtype=torch.float, device=device).view(-1, 1)
D_optim = optim.SGD(params=D.parameters(), lr=learning_rate)
G_optim = optim.SGD(params=G.parameters(), lr=learning_rate)



score_tr = np.zeros((train_epoch, 4*7+3))
s = metric.compute_score_raw('figr', length, data_dir, 200, 16, 'Runs/WGAN_FIGR_Results/real/','Runs/WGAN_FIGR_Results/fake/',
                             G,z_shape, conv_model='inception_v3', workers=4)
score_tr[0] = s
np.save('%s/score_tr.npy' % ('Runs/WGAN_FIGR_Results'), score_tr)


# starting training
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
        G_optim.param_groups[0]['lr'] /= 10
        D_optim.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optim.param_groups[0]['lr'] /= 10
        D_optim.param_groups[0]['lr'] /= 10
        print("learning rate change!")


    ### WGAN-GP
    ### training discriminator first: [discriminator_pred, discriminator_targets]
    ### training generator then; [output, generator_targets]
    print('starting triaing')
    num_iter = 0
    epoch_start_time = time.time()
    print(len(train_loader))
    for x_, _ in train_loader:
        if np.shape(x_)[0]!=batch_size:
            continue

        ## training discriminator ##
        G.train()
        #x_ = torch.from_numpy(x_)
        x_=x_.type(torch.FloatTensor)
        x_ = normalize_data(x_)
        real_batch = Variable(x_.to(device))
        # [32, 100]
        # [32, 3, 64, 64]
        # [32, 3, 64, 64]
        fake_batch = G(torch.tensor(np.random.normal(size=(batch_size, z_shape)), dtype=torch.float, device=device))
        training_batch = torch.cat([real_batch, fake_batch])

        

        # Training discriminator
        gradient_penalty = calc_gradient_penalty(D, real_batch, fake_batch)
        discriminator_pred = D(training_batch)
        # [64, 3, 64, 64]
        # [64, 1]
        discriminator_loss = wassertein_loss(discriminator_pred, discriminator_targets)
        discriminator_loss += gradient_penalty
        D_losses.append(discriminator_loss)

        D_optim.zero_grad()
        discriminator_loss.backward()
        D_optim.step()

        ## Training generator ##
        output = D(G(torch.tensor(np.random.normal(size=(batch_size, z_shape)), dtype=torch.float, device=device)))
        generator_loss = wassertein_loss(output, generator_targets)
        G_losses.append(generator_loss)

        G_optim.zero_grad()
        generator_loss.backward()
        G_optim.step()
    
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time


    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),                                                          torch.mean(torch.FloatTensor(G_losses))))
    p = 'Runs/WGAN_FIGR_Results/' + str(epoch + 1) + '.png'
    if not os.path.isdir('Runs/WGAN_FIGR_Results'):
        os.mkdir('Runs/WGAN_FIGR_Results')
    writer = SummaryWriter('Runs/WGAN_FIGR_Results')
    WGAN_show_result((epoch+1), G, p, z_shape, batch_size, device, writer)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    ## evaluation
    s = metric.compute_score_raw('figr', length, data_dir, 200, 16, 'Runs/WGAN_FIGR_Results/real/','Runs/WGAN_FIGR_Results/fake/',
                                 G,z_shape, conv_model='inception_v3', workers=4)
    score_tr[epoch] = s

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G.state_dict(), "Runs/WGAN_FIGR_Results/generator_param.pkl")
torch.save(D.state_dict(), "Runs/WGAN_FIGR_Results/discriminator_param.pkl")
with open('Runs/WGAN_FIGR_Results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path='Runs/WGAN_FIGR_Results/train_hist.png')

'''
images = []
for e in range(train_epoch):
    img_name = 'Runs/WGAN_FIGR_Results/' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('Runs/WGAN_FIGR_Results/generation_animation.gif', images, fps=5)
'''