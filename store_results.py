import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import types
import math
import numpy as np
from PIL import Image
import PIL.Image as pil
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import PerceptualSimilarity.lpips.lpips as lpips
import glob
import gc
import pytorch_msssim.pytorch_msssim as pytorch_msssim
from  torch.cuda.amp import autocast
from model import BRViT
import matplotlib.pyplot as plt
from pytorch_ssim.pytorch_ssim import ssim
from torchvision.utils import save_image

import pandas as pd
from tqdm import tqdm
import sys
import cv2
import time
import scipy.misc

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


feed_width = 1536
feed_height = 1024


#to store results
def store_results():
    device = torch.device("cuda")
    
    model = BRViT().to(device)

    PATH = "weights/BRViT_53_0.pt"
    model.load_state_dict(torch.load(PATH), strict=True)
    
    with torch.no_grad():
        for i in tqdm(range(4400,4694)):
            csv_file = "Bokeh_Data/test.csv"
            data = pd.read_csv(csv_file)
            root_dir = "."
            idx = i - 4400
            bok = pil.open(root_dir + data.iloc[idx, 0][1:]).convert('RGB')
            input_image = pil.open(root_dir + data.iloc[idx, 1][1:]).convert('RGB')
            original_width, original_height = input_image.size

            org_image = input_image
            org_image = transforms.ToTensor()(org_image).unsqueeze(0)

            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # prediction
            org_image = org_image.to(device)
            input_image = input_image.to(device)

            bok_pred = model(input_image)

            bok_pred = F.interpolate(bok_pred,(original_height,original_width),mode = 'bilinear')
            
            save_image(bok_pred,'Results/'+ str(i) +'.png')



if __name__ == "__main__":
    store_results()


