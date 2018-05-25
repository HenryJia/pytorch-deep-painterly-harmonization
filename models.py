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


# An attempt at an efficient patch matching algorithm
# Important Note: This function is NOT differentiable with respect to x or the mask
def patch_match(x, y, mask, patch_size = 3, radius = 3, stride = 1):
    batch, channels, height, width = x.size()

    y_pad = F.pad(y, (radius // 2, radius // 2, radius // 2, radius // 2)) # Left, right, up, down
    distance_all = []
    for i in range(0, radius, stride): # Searching/matching in row-major order
        for j in range(0, radius, stride):
            distance_pix = torch.sum((y_pad[:, :, i:i + height, j:j + width] - x) ** 2, dim = 1, keepdim = True)
            distance_all += [F.avg_pool2d(distance_pix, patch_size, stride = 1, padding = patch_size // 2)]

    distance_all = torch.cat(distance_all, dim = 1) # Thus this stack of distances will be in row major order
    location_min = torch.argmin(distance_all, dim = 1) # get the pixel/patch with the minimal distance
    location_min = location_min * mask # Only need to match within the mask
    distance_min_x = torch.fmod(location_min, radius) - radius // 2 # Need to adjust to take into account searching behind
    distance_min_y = location_min / radius - radius // 2

    grid_x = torch.arange(width).cuda().unsqueeze(0).unsqueeze(0) + distance_min_x # Make our grid and use PyTorch's grid_sample
    grid_y = torch.arange(height).cuda().unsqueeze(1).unsqueeze(0) + distance_min_y
    grid_x = torch.clamp(grid_x.float() / width, 0, 1) * 2 - 1
    grid_y = torch.clamp(grid_y.float() / height, 0, 1) * 2 - 1

    grid = torch.stack([grid_x, grid_y], dim = 3) # put the grids together
    #print(grid.size(), distance_all.size(), location_min.size(), distance_min_x.size(), distance_min_y.size())

    # Now I know PyTorch uses bilinear for this whereas Deep Painterly Harmonisation uses nearest neighbour sampling, but since our indices are all integers
    # it makes no difference, though bilinear is much more compute intensive, but we have GPUs so it shouldn't matter too much
    out = F.grid_sample(y, grid)
    return out
