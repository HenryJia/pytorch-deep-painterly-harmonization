from collections import namedtuple

import torch
import torch.nn.functional as F
from torchvision import models


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad = False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained = True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 26):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        out = {'relu1_2' : h_relu1_2, 'relu2_2' : h_relu2_2, 'relu3_1' : h_relu3_3, 'relu4_1' : h_relu4_3, 'relu5_1' : h_relu5_3}
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad = False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained = True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)
        out = {'relu1_2' : h_relu1_2, 'relu2_2' : h_relu2_2, 'relu3_1' : h_relu3_4, 'relu4_1' : h_relu4_4, 'relu5_1' : h_relu5_4}
        return out


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def patchdot(x, y, patch_size = 3): # patchwise dot product
    dot = torch.sum(x * y, dim = 1, keepdim = True)
    norm = F.avg_pool2d(dot, patch_size, stride = 1, padding = patch_size // 2).squeeze(1) * patch_size ** 2
    return norm

def cosine_distance(x, y, patch_size = 3):
    return patchdot(x, y, patch_size) / (patchdot(y, y, patch_size) * patchdot(x, x, patch_size))


# An attempt at an efficient patch matching algorithm
# Important Note: This function is NOT differentiable
def patch_match(x, y, mask, patch_size = 3, stride = 1):
    batch, channels, height, width = x.size()

    distance_min = torch.ones(batch, height, width).cuda().detach() * 1e6
    grid_x = torch.zeros(batch, height, width).cuda().detach()
    grid_y = torch.zeros(batch, height, width).cuda().detach()
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            distance = cosine_distance(y[:, :, i:, j:].detach(), x[:, :, :-i or None, :-j or None].detach(), patch_size)

            #if i > 0 or j > 0:
                #distance = F.pad(distance, (0, j, 0, i)) # (left, right, up down)
            is_min = (distance < distance_min[:, :-i or None, :-j or None]).float()
            distance_min[:, :-i or None, :-j or None] = is_min * distance + (1 - is_min) * distance_min[:, :-i or None, :-j or None]
            grid_x[:, :-i or None, :-j or None] = is_min * j + (1 - is_min) * grid_x[:, :-i or None, :-j or None]
            grid_y[:, :-i or None, :-j or None] = is_min * i + (1 - is_min) * grid_y[:, :-i or None, :-j or None]

    grid_x = mask * grid_x.detach() + (1 - mask) * torch.arange(width).cuda().float().unsqueeze(0).unsqueeze(0)
    grid_y = mask * grid_y.detach() + (1 - mask) * torch.arange(height).cuda().float().unsqueeze(-1).unsqueeze(0)
    grid_x = torch.clamp(grid_x / (width - 1), 0, 1) * 2 - 1
    grid_y = torch.clamp(grid_y / (height - 1), 0, 1) * 2 - 1

    grid = torch.stack([grid_x, grid_y], dim = 3) # put the grids together
    #print(grid.size(), distance_all.size(), location_min.size(), distance_min_x.size(), distance_min_y.size())

    # Now I know PyTorch uses bilinear for this whereas Deep Painterly Harmonisation uses nearest neighbour sampling, but since our indices are all integers
    # it makes no difference, though bilinear is much more compute intensive, but we have GPUs so it shouldn't matter too much
    out = F.grid_sample(y, grid)
    return out
