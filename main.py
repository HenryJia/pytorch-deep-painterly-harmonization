import sys
import time

import numpy as np
from scipy.misc import imread
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam, LBFGS
from torchvision.transforms import ToTensor, Normalize, Compose

import pylab as plt
plt.ion()

from tqdm import tqdm
tqdm.monitor_interval = 0

from models import Vgg19, gram_matrix, patch_match
from data_utils import read_img

if len(sys.argv) != 5:
    print('Syntax: {} <style.jpg> <naive.jpg> <mask.jpg> <out.jpg>'.format(sys.argv[0]))
    sys.exit(0)

(style_fn, naive_fn, mask_fn, out_fn) = sys.argv[1:]

transform = Compose([ToTensor(), Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

style_img = transform(Image.open(style_fn)).unsqueeze(0).cuda()
naive_img = transform(Image.open(naive_fn)).unsqueeze(0).cuda()
mask_img = imread(mask_fn)[..., 0].astype(np.float32)
tmask_img = gaussian_filter(mask_img, sigma = 3)
tmask_img = torch.from_numpy(tmask_img).unsqueeze(0).cuda() / 255.0
mask_img = torch.from_numpy(mask_img).unsqueeze(0).cuda() / 255.0

naive_img.requires_grad_(True)
naive_img_original = naive_img.clone()

net = Vgg19().cuda()
#optimizer = Adam([naive_img], lr = 1e1)
optimizer = LBFGS([naive_img], max_iter = 1000)

features_style = net(style_img)
features_naive_original = net(naive_img)

layers_content = ['relu4_1']
layers_style = ['relu3_1', 'relu4_1', 'relu5_1']

features_style_nearest = {}
for l in layers_style:
    features_style_nearest[l] = patch_match(features_naive_original[l], features_style[l], patch_size = 3, radius = 1)
#features_style_nearest = features_style

for i in tqdm(range(1)):
    def closure():
        features_naive = net(naive_img)

        loss_content = 0
        mask = mask_img
        for l in layers_content:
            while (mask.size(1) != features_naive[l].size(2)):
                mask = F.avg_pool2d(mask, 2)
            loss_content += torch.mean(mask * (features_naive[l] - features_naive_original[l].detach()) ** 2)

        loss_style = 0
        mask = mask_img
        for l in layers_style:
            while (mask.size(1) != features_naive[l].size(2)):
                mask = F.avg_pool2d(mask, 2)
            gram_naive = gram_matrix(mask * features_naive[l])
            gram_style = gram_matrix(mask * features_style_nearest[l])
            loss_style += torch.mean((gram_naive - gram_style.detach()) ** 2)

        loss_variation = torch.mean((naive_img[:, :, 1:] - naive_img[:, :, :-1]) ** 2) + torch.mean((naive_img[:, :, :, 1:] - naive_img[:, :, :, :-1]) ** 2)

        loss = 5 * loss_content + 100 * loss_style + 150 * loss_variation
        net.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        naive_img.grad.data *= mask_img
        return loss

    optimizer.step(closure)

    naive_img = tmask_img * naive_img + (1 - tmask_img) * naive_img_original
    out = np.transpose(naive_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    #out = (out - np.max(out)) / (np.max(out) - np.min(out))
    plt.figure()
    plt.imshow(out)
