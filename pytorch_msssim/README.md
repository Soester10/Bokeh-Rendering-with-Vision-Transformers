# pytorch-msssim

### Differentiable Multi-Scale Structural Similarity (SSIM) index 

This small utiliy provides a differentiable MS-SSIM implementation for PyTorch based on Po Hsun Su's implementation of SSIM @ https://github.com/Po-Hsun-Su/pytorch-ssim.
At the moment only the product method for MS-SSIM is supported.

## Installation

Master branch now only supports PyTorch 0.4 or higher. All development occurs in the dev branch (`git checkout dev` after cloning the repository to get the latest development version).

To install the current version of pytorch_mssim:

1. Clone this repo.
2. Go to the repo directory.
3. Run `python setup.py install`

or 

1. Clone this repo.
2. Copy "pytorch_msssim" folder in your project.

To install a version of of pytorch_mssim that runs in PyTorch 0.3.1 or lower use the tag checkpoint-0.3. To do so, run the following commands after cloning the repository:

```
git fetch --all --tags
git checkout tags/checkpoint-0.3
```

## Example

### Basic usage
```python
import pytorch_msssim
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = pytorch_msssim.MSSSIM()

img1 = torch.rand(1, 1, 256, 256)
img2 = torch.rand(1, 1, 256, 256)

print(pytorch_msssim.msssim(img1, img2))
print(m(img1, img2))


```

### Training

For a detailed example on how to use msssim for optimization, take a look at the file max_ssim.py.


### Stability and normalization

MS-SSIM is a particularly unstable metric when used for some architectures and may result in NaN values early on during the training. The msssim method provides a normalize attribute to help in these cases. There are three possible values. We recommend using the value normalized="relu" when training. 

- None : no normalization method is used and should be used for evaluation
- "relu" : the `ssim`and `mc` values of each level during the calculation are rectified using a relu ensuring that negative values are zeroed
- "simple" : the `ssim`result of each iteration is averaged with 1 for an expected lower bound of 0.5 - should ONLY be used for the initial iterations of your training or when averaging below 0.6 normalized score

Currently and due to backward compability, a value of True will equal the "simple" normalization.

## Reference
https://ece.uwaterloo.ca/~z70wang/research/ssim/

https://github.com/Po-Hsun-Su/pytorch-ssim

Thanks to z70wang for proposing MS-SSIM and providing the initial implementation, and Po-Hsun-Su for the initial differentiable SSIM implementation for Pytorch. 
