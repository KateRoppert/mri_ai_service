import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import morphology
from medpy import metric as medmetric


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss

def dice_coef_loss(input, target):
    smooth = 1e-7
    intersection = -2. * (target * input).sum() + smooth
    union = target.sum() + input.sum() + smooth
    # print ("intersection and union:", intersection, union)
    return (intersection / union)


def bce_dice_loss(input, target):
    dicescore = dice_coef_loss(input, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(input, target)
    return 0.5*bceloss + 0.5*dicescore, bceloss, dicescore


def bce_focal_dice_loss(input, target):
    dice_loss = dice_coef_loss(input, target)
    focal = FocalLoss()
    bce = nn.BCELoss()
    bce_loss = bce(input, target)
    focal_loss = focal(input, target)

    return 0.3*bce_loss + 0.5*focal_loss + 0.2*dice_loss, bce_loss, focal_loss, dice_loss

######################### metrics #####################

def dice_coef_metric_batch(outputs, targets):
    scores = []
    for i in range(outputs.shape[0]):
        output = outputs[i]
        target = targets[i]
        intersection = 2.0 * (target * output).sum()
        union = target.sum() + output.sum()
        if target.sum() == 0 and output.sum() == 0:
            scores.append(1.0)
        else:
            scores.append(intersection / union)
    return np.mean(scores)


def dice_coef_metric_batch_fair(outputs, targets):
    scores = []
    for i in range(outputs.shape[0]):
        output = outputs[i]
        target = targets[i]
        intersection = 2.0 * (target * output).sum()
        union = target.sum() + output.sum()
        if target.sum() or output.sum():
            scores.append(intersection / union)
    return np.mean(scores)


def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union



def dice_coef_metric_3d(outputs, targets):
    print(outputs.shape[0], targets.shape[0])
    scores = []
    intersection, union = [], []
    for i in range(outputs.shape[0]):
        output = outputs[i]
        target = targets[i]
        try:
            if output.sum() or target.sum():
                intersection.append(2.0 * (target * output).sum())
                union.append(target.sum() + output.sum())
        except ZeroDivisionError:
            dice_scores.append(0.0)
    return np.sum(intersection)/np.sum(union)

############# temp for sanity check #############

def dice_coef_metric_batch_sanity(outputs, targets):
    scores, scores_exist = [], []
    for i in range(outputs.shape[0]):
        output = outputs[i]
        target = targets[i]
        intersection = 2.0 * (target * output).sum()
        union = target.sum() + output.sum()
        if target.sum() == 0:
            scores.append(1.0)
        else:
            scores.append(intersection / union)

        if target.sum():
            scores_exist.append(intersection / union)
    return np.mean(scores), scores_exist

def dice_coef_metric_batch_2outs(outputs, targets):
    scores = []
    for i in range(outputs.shape[0]):
        output = outputs[i]
        target = targets[i]
        intersection = 2.0 * (target * output).sum()
        union = target.sum() + output.sum()
        if target.sum() == 0 and output.sum() == 0:
            scores.append(1.0)
        else:
            scores.append(intersection / union)
    return np.mean(scores), scores
#################################################

def calc_recall(input, target):
    intersection = ((target*input).sum())
    return (intersection/target.sum())


def calc_precision(input, target):
    intersection = ((target*input).sum())
    return (intersection/input.sum())


def calc_ravd(input, target):
    return medmetric.binary.ravd(input,target)


def calc_assd(input, target):
    return medmetric.binary.assd(input,target)


def calc_asd(input, target):
    return medmetric.binary.asd(input,target)


def calc_hd(input, target):
    return medmetric.binary.hd(input,target)

    
def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1^morphology.binary_erosion(input_1, conn)
    Sprime = input_2^morphology.binary_erosion(input_2, conn)
    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds







