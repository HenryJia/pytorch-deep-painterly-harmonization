import numpy as np

from scipy.misc import imread

def read_img(fn):
    return np.transpose(imread(fn).astype(np.float32), (2, 0, 1))
