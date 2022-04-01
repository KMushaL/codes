'''

Author: wl

Date: 2022-03-31 20:07:00

LastEditTime: 2022-04-01 10:59:20

LastEditors: Please set LastEditors

Description: mixup算法的实现

FilePath: \codes\models\mixup.py

'''

import torch
import numpy as np


def Pad(x1, x2):
    if x1.shape[0] < x2.shape[0]:
        pad = torch.zeros_like(x2)[:x2.shape[0] - x1.shape[0]]
        x1 = torch.cat((x1, pad), 0)
    elif x1.shape[0] > x2.shape[0]:
        pad = torch.zeros_like(x1)[:x1.shape[0] - x2.shape[0]]
        x2 = torch.cat((x2, pad), 0)
        
    return x1, x2


def Mixup(bundle1, bundle2):
    """
    bundle1和bundle2都各有两个样本（若是训练模式）：正样本和负样本，且都是一套服装
    """
    alpha = 0.2
    lam = np.random.beta(alpha, alpha)

    for sample1, sample2 in zip(bundle1, bundle2):
        x1, x2 = Pad(sample1.x, sample2.x)
        y1, y2 = sample1.y, sample2.y
        sample1.x = lam * x1 + (1 - lam) * x2
        sample1.y = lam * y1 + (1 - lam) * y2

    return bundle1
