import math
import numpy as np
import cv2
import yaml


def read_model_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def bchw_2_blc(x):
    B, C, H, W = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)
    return x


def blc_2_bchw(x):
    B, L, C = x.shape
    H = W = int(math.sqrt(L))
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return x


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def read_img(filename):
    img = cv2.imread(filename)
    return img[:, :, ::-1].astype('float32') / 255.0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
