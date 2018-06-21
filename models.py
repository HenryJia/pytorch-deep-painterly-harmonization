from collections import namedtuple

import torch
import torch.nn.functional as F
from torchvision import models

import time


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad = False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained = True).features
        self.features = vgg_pretrained_features
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
        self.features = vgg_pretrained_features
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
    gram = features.bmm(features_t)
    return gram


def patchdot(x, y, patch_size = 3): # patchwise dot product
    dot = torch.sum(x * y, dim = 1, keepdim = True)
    norm = F.avg_pool2d(dot, patch_size, stride = 1, padding = patch_size // 2).squeeze(1) / patch_size ** 2
    return norm

def cosine_similarity(x, y, patch_size = 3):
    out = patchdot(x, y, patch_size)
    out = out / (torch.sqrt(patchdot(y, y, patch_size) * patchdot(x, x, patch_size)) + 1e-8)
    return out


# An attempt at an efficient patch matching algorithm
# Important Note: This function is NOT differentiable
# Not working, needs fixing
def patch_match(x, y, patch_size = 3, stride = 1):
    batch, channels, height_x, width_x = x.size()
    batch, channels, height_y, width_y = y.size()

    with torch.no_grad():
        similarity_max = torch.zeros(batch, height_x, width_x).cuda() - 1
        grid_j = torch.zeros(batch, height_x, width_x).cuda().float()
        grid_i = torch.zeros(batch, height_x, width_x).cuda().float()

        y_pad = torch.cat([y, y], dim = 2)
        y_pad = torch.cat([y_pad, y_pad], dim = 3)

        for i in range(0, height_y, stride): # Searching/matching in row-major order
            for j in range(0, width_y, stride):
                similarity = cosine_similarity(y_pad[:, :, i:i + height_x, j:j + width_x], x)

                is_max = (similarity > similarity_max).float()
                similarity_max = is_max * similarity + (1 - is_max) * similarity_max
                grid_j = is_max * j + (1 - is_max) * grid_j
                grid_i = is_max * i + (1 - is_max) * grid_i

        grid_j = grid_j + torch.arange(width_y).cuda().float().unsqueeze(0).unsqueeze(0)
        grid_i = grid_i + torch.arange(height_y).cuda().float().unsqueeze(-1).unsqueeze(0)

        grid_j = grid_j % width_y
        grid_i = grid_i % height_y

        grid_j = grid_j.float() / (width_y - 1) * 2 - 1
        grid_i = grid_i.float() / (height_y - 1) * 2 - 1

        grid = torch.stack([grid_j, grid_i], dim = 3) # put the grids together
        #print(grid.size(), similarity_all.size(), location_min.size(), similarity_min_x.size(), similarity_min_y.size())

        # Now I know PyTorch uses bilinear for this whereas Deep Painterly Harmonisation uses nearest neighbour sampling, but since our indices are all integers
        # it makes no difference, though bilinear is much more compute intensive, but we have GPUs so it shouldn't matter too much
        out = F.grid_sample(y, grid, mode = 'nearest')
        return out

# We disable gradients for speed
def downsampling(x, size = None, scale_factor = None, mode = 'bilinear'):
    with torch.no_grad():
        # define size if user has specified scale_factor
        if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
        h = torch.arange(0,size[0]).float() / (size[0]-1) * 2 - 1
        w = torch.arange(0,size[1]).float() / (size[1]-1) * 2 - 1
        # create grid
        grid = torch.zeros(size[0],size[1],2)
        grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
        grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
        # expand to match batch size
        grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
        if x.is_cuda: grid = grid.cuda()
        # do sampling
        return F.grid_sample(x, grid, mode = mode)
