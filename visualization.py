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


def DCGAN_show_result(num_epoch, G ,path, image_number, z_shape):
    z_ = torch.randn((image_number*image_number, z_shape)).view(-1, z_shape, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
    G.train()
   

    size_figure_grid = image_number
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(image_number, image_number))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(image_number*image_number):
        i = k // image_number
        j = k % image_number
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)



def WGAN_show_result(num_epoch, G ,path, z_shape, batch_size, device, writer):
    def unnormalize_data(data):
        data += 1
        data /= 2
        return data
    G.eval()
    with torch.no_grad():
        img = G(torch.tensor(np.random.normal(size=(batch_size * 3, z_shape)), dtype=torch.float, device=device))
    img = img.detach().cpu().numpy()
    img = np.concatenate([np.concatenate([img[i * 3 + j] for j in range(3)], axis=-2) for i in range(batch_size)], axis=-1)
    img = unnormalize_data(img)
    # (1, 96, 1024)
    # img = np.concatenate([training_images, img], axis=-2)
    writer.add_image('Validation_generated', img, num_epoch)













def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()