import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np



#### data preparation for WGAN: single channel image preprocessing
### complex for single channel image
def find_classes(root_dir):
    retour=[]
    for (root,dirs,files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r=root.split('/')
                lr=len(r)
                retour.append((f,r[lr-2]+"/"+r[lr-1],root))
    print("== Found %d items "%len(retour))
    return retour

def index_classes(items):
    idx={}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]]=len(idx)
    print("== Found %d classes"% len(idx))
    return idx

class FIGR(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.all_items=find_classes(self.root)
        self.idx_classes=index_classes(self.all_items)

    def __getitem__(self, index):
        filename=self.all_items[index][0]
        img=str.join('/',[self.all_items[index][2],filename])

        target=self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return  img,target

    def __len__(self):
        return len(self.all_items)


def WGAN_data_preparation(data_dir,length,channels,batch_size):

	train_loader_1 = FIGR(data_dir, transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
	                                                               lambda x: x.resize((length,length)),
	                                                               lambda x: np.reshape(x, (channels, length, length)), 

	                                                               ]))
	train_loader = torch.utils.data.DataLoader(train_loader_1, batch_size=batch_size, shuffle=True)
	return train_loader





##### data preprocess for DCGAN
##### simple for three channel images
def DCGAN_data_preparation(data_dir,length,batch_size):
	transform = transforms.Compose([
	        transforms.Scale(length),
	        transforms.ToTensor(),
	        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	dset = datasets.ImageFolder(data_dir, transform)
	train_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle=True)
	return train_loader


def WGAN_data_evaluation(data_dir,length,channels=1):
	dset = FIGR(data_dir, transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
	                                                               lambda x: x.resize((length,length)),
	                                                               lambda x: np.reshape(x, (channels, length, length)), 

	                                                               ]))
	return dset


def DCGAN_data_evaluation(data_dir,length):
	transform = transforms.Compose([
	        transforms.Scale(length),
	        transforms.ToTensor(),
	        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	dset = datasets.ImageFolder(data_dir, transform)
	return dset