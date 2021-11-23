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
from PerceptualSimilarity.util import util


import pandas as pd
from tqdm import tqdm
import sys
import cv2

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

batch_size = 1
feed_width = 1536
feed_height = 1024



def evaluate():
    tot_lpips_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    device = torch.device("cuda")
    loss_fn = lpips.LPIPS(net='alex').to(device)

    for i in tqdm(range(294)):
        csv_file = "Bokeh_Data/test.csv"
        root_dir = "."
        dataa = pd.read_csv(csv_file)
        idx = i

        img0 = util.im2tensor(util.load_image(root_dir + dataa.iloc[idx, 0][1:])) # RGB image from [-1,1]
        img1 = util.im2tensor(util.load_image(f"Results/{4400+i}.png"))
        img0 = img0.to(device)
        img1 = img1.to(device)

        lpips_loss = loss_fn.forward(img0, img1)
        tot_lpips_loss += lpips_loss.item()


        total_psnr += compare_psnr(I0,I1)
        total_ssim += compare_ssim(I0, I1, multichannel=True)


    print("TOTAL LPIPS:",":", tot_lpips_loss / 294)
    print("TOTAL PSNR",":", total_psnr / 294)
    print("TOTAL SSIM",":", total_ssim / 294)


if __name__ == "__main__":
    evaluate()


