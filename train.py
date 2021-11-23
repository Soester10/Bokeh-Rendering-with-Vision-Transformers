from __future__ import absolute_import, division, print_function
import cv2
import numbers
import math
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from pytorch_msssim.pytorch_msssim import msssim
import torchvision
from torch.autograd import Variable
from pytorch_ssim.pytorch_ssim import ssim
import PerceptualSimilarity.lpips.lpips as lpips

from tqdm import tqdm

device = torch.device("cuda:0")

from model import BRViT


feed_width = 768
feed_height =  512


bokehnet = BRViT().to(device)
batch_size = 1


PATH = "BRViT_53_0.pt"
bokehnet.load_state_dict(torch.load(PATH), strict=False)

print("weights loaded!!")


class bokehDataset(Dataset):
    
    def __init__(self, csv_file,root_dir, transform=None):
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        bok = pil.open(self.root_dir + self.data.iloc[idx, 0][1:]).convert('RGB')
        org = pil.open(self.root_dir + self.data.iloc[idx, 1][1:]).convert('RGB')
            
        bok = bok.resize((feed_width, feed_height), pil.LANCZOS)
        org = org.resize((feed_width, feed_height), pil.LANCZOS)
        if self.transform : 
            bok_dep = self.transform(bok)
            org_dep = self.transform(org)
        return (bok_dep, org_dep)

transform1 = transforms.Compose(
    [
    transforms.ToTensor(),
])


transform2 = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
])


transform3 = transforms.Compose(
    [
    transforms.RandomVerticalFlip(p=1),
    transforms.ToTensor(),
])



trainset1 = bokehDataset(csv_file = './Bokeh_Data/train.csv', root_dir = '.',transform = transform1)
trainset2 = bokehDataset(csv_file = './Bokeh_Data/train.csv', root_dir = '.',transform = transform2)
trainset3 = bokehDataset(csv_file = './Bokeh_Data/train.csv', root_dir = '.',transform = transform3)


trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([trainset1,trainset2,trainset3]), batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = bokehDataset(csv_file = './Bokeh_Data/test.csv',  root_dir = '.', transform = transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)


learning_rate = 0.00001

optimizer = optim.Adam(bokehnet.parameters(), lr=learning_rate, betas=(0.9, 0.999))
                                                            

sm = nn.Softmax(dim=1)

MSE_LossFn = nn.MSELoss()
L1_LossFn = nn.L1Loss()


def train(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0
    running_sal_loss = 0
    running_loss = 0

    for i,data in enumerate(dataloader,0) :
        bok ,  org = data
        bok ,  org = bok.to(device) , org.to(device)

        optimizer.zero_grad()
        
        bok_pred = bokehnet(org)

        loss = (1-msssim(bok_pred, bok))

##        loss = L1_LossFn(bok_pred, bok)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        if (i % 10 == 0):
            print ('Batch: ',i,'/',len(dataloader),' Loss:', loss.item())
        

        if ((i+1)%8000==0):
            torch.save(bokehnet.state_dict(), './weights/BRViT_' + str(epoch) + '_' + str(i) + '.pt')
            print(loss.item())
 
    print (running_loss/len(dataloader))




def val(dataloader):
    running_l1_loss = 0
    running_ms_loss = 0
    running_lips_loss = 0

    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader),0) : 
            bok , org = data 
            bok , org = bok.to(device) , org.to(device)
            
            bok_pred = bokehnet(org)

            try:
                l1_loss = L1_LossFn(bok_pred, bok)
            except:
                l1_loss = L1_LossFn(bok_pred[0], bok)

            ms_loss = 1-ssim(bok_pred, bok)
            

            running_l1_loss += l1_loss.item()
            running_ms_loss += ms_loss.item()


    print ('Validation l1 Loss: ',running_l1_loss/len(dataloader))   
    print ('Validation ms Loss: ',running_ms_loss/len(dataloader))

##    torch.save(bokehnet.state_dict(), './weights/BRViT_'+str(epoch)+'.pt')

    try:
        with open("log.txt", 'a') as f:
            f.write(f"{running_ms_loss/len(dataloader)}\n")
    except:
        pass



start_ep = 0
for epoch in range(start_ep, 40):
    print (epoch)
   
    train(trainloader)

    with torch.no_grad():
        val(testloader)
