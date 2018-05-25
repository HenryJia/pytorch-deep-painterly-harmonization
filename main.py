import sys

import numpy as np
from scipy.misc import imread
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
mask_img = torch.from_numpy(np.transpose(imread(mask_fn).astype(np.float32), (2, 0, 1))).unsqueeze(0).cuda() / 255.0

# We only want the max and not the argmax so only take the 0th output
mask_img = torch.max(torch.ceil(mask_img), dim = 1)[0]
naive_img.requires_grad_(True)

net = Vgg19().cuda()
#optimizer = Adam([naive_img], lr = 1e1)
optimizer = LBFGS([naive_img], max_iter = 100)

features_style = net(style_img)

layers_content = ['relu4_1']
layers_style = ['relu3_1', 'relu4_1', 'relu5_1']

for i in tqdm(range(10)):
    def closure():
        features_naive = net(naive_img)

        mask = mask_img.float()
        features_style_nearest = {}
        for l in layers_style:
            while (mask.size(1) != features_naive[l].size(2)):
                mask = torch.round(F.avg_pool2d(mask.float(), 2))
            mask = mask.long()
            #mask = torch.round(F.avg_pool2d(mask_img, round(mask_img.size(2) / features_naive[l].size(2)))).long()
            features_style_nearest[l] = patch_match(features_naive[l], features_style[l], mask, patch_size = 3, radius = min(mask.shape[2:]))
        #features_style_nearest = features_style

        loss = 0
        mask = mask_img.float()
        for l in layers_content:
            while (mask.size(1) != features_naive[l].size(2)):
                mask = F.avg_pool2d(mask.float(), 2)
            loss += 5 * torch.mean(mask * (features_naive[l] - features_style[l].detach()) ** 2)
        mask = mask_img.float()
        for l in layers_style:
            while (mask.size(1) != features_naive[l].size(2)):
                mask = F.avg_pool2d(mask.float(), 2)
            loss += 100 * torch.mean((gram_matrix(mask * features_naive[l]) - gram_matrix(mask * features_style_nearest[l]).detach()) ** 2)

        net.zero_grad()
        optimizer.zero_grad()

        loss.backward()
        return loss

    optimizer.step(closure)

    out = np.transpose(naive_img.detach().squeeze().cpu().numpy(), (1, 2, 0))
    #out = (out - np.max(out)) / (np.max(out) - np.min(out))
    plt.figure()
    plt.imshow(out)
