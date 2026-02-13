__author__ = "bobbqe"

from glob import glob
import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def select_worst_checkpoint(folder, fold, model):
    fns = sorted(glob(osp.join(folder, "*"+model+"_fold"+str(fold)+"*.pth")))
    if len(fns) >= 1:
        return fns[0]
    else:
        print('emplty {}'.format(folder))
        return None

def select_best_checkpoint(folder, fold, model):
    fns = sorted(glob(osp.join(folder, "*"+model+"_fold"+str(fold)+"*.pth")))
    if len(fns) >= 1:
        return fns[-1]
    else:
        print('emplty {}'.format(folder))
        return None

def select_notbest_checkpoint(folder, fold, model):
    fns = sorted(glob(osp.join(folder, "*"+model+"_fold"+str(fold)+"*.pth")))
    if len(fns) >= 1:
        return fns[10]
    else:
        print('emplty {}'.format(folder))
        return None

def select_3best_checkpoint(folder, fold, model):
    fns = sorted(glob(osp.join(folder, "*"+model+"_fold"+str(fold)+"*.pth")))
    if len(fns) >= 1:
        return fns[-1], fns[-2], fns[-3]
    else:
        print('emplty {}'.format(folder))
        return None

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

    raise ValueError("Unsupported input type" + str(type(x)))
