__author__ = "bobbqe"

import argparse
import random
import time
import os
import glob
import csv
import math
import distutils
import numpy as np
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
from torchvision import models
import nibabel as nib

from dataset_reader import *
from pretrained_models import *
from utils import *
from loss_metric_multiclass import *


SEED = 777
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
os.environ["XDG_CACHE_HOME"] = "./model_cache/"

random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

if torch.cuda.is_available:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
else:
    print("ERROR: CUDA is not available. Exit")

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#
# ensemble_models_3d = []
# path_to_3d_models = '/outs_multi/checkpoints_UneXt50_mltlbl_loss_full321'
# import os
# with torch.set_grad_enabled(False):
#     for model_path in os.listdir(path_to_3d_models):
#         model = torch.load(path_to_3d_models + "/" + model_path)
#         model = model.cuda()
#         model = model.eval()
#         print("Add to 3D ensemble: ", model_path)
#         ensemble_models_3d.append(model)
#

# MODELS = {
#     "unet_resnet18": UNet_class(models.resnet18(pretrained=True), num_classes=5),
#     "unet_resnet34": UNet_class(models.resnet34(pretrained=True), num_classes=5),
#     "linknet_resnet18": LinkNet_class(models.resnet18(pretrained=True), num_classes=5),
#     "linknet_resnet34": LinkNet_class(models.resnet34(pretrained=True), num_classes=5),
#     "linknet_seresnet50": SE_class_LinkNet_network(se_resnet50(),num_classes=5),
#     "linknet_seresnet101": SE_class_LinkNet_network(se_resnet101(),num_classes=5),
#     "linknet_seresnext50": SE_class_LinkNet_network(se_resnext50_32x4d(),num_classes=5),
#     "linknet_seresnext101": SE_class_LinkNet_network(se_resnext101_32x4d(),num_classes=5),
#     "linknet_senet154": SE_class_LinkNet_network(senet154(),num_classes=5),
#     "linknet_seresnet152": SE_class_LinkNet_network(se_resnet152(),num_classes=5),
#     ### ResNet-D version with more gentil first convolution -> changed to the set of 3 Conv2D 3x3
#     "linknet_seresnet50D": SE_class_LinkNet_network(se_resnet50D(pretrained=None),num_classes=5),
#     "linknet_seresnet101D": SE_class_LinkNet_network(se_resnet101D(pretrained=None),num_classes=5),
#     "linknet_seresnext50D": SE_class_LinkNet_network(se_resnext50D_32x4d(pretrained=None),num_classes=5),
#     "linknet_seresnext101D": SE_class_LinkNet_network(se_resnext101D_32x4d(pretrained=None),num_classes=5),
#     "unet_resnet34D": UNetD_class(models.resnet34(pretrained=False), num_classes=5),
#     "unet_resnet50D": UNetD_class(models.resnet50(pretrained=False), num_classes=5),
#     "psp_seresnext50": smp.PSPNet(encoder_name='se_resnext50_32x4d', encoder_weights='imagenet', classes=5, activation='logsoftmax'), #
#     "fpn_seresnext50": smp.FPN(encoder_name='se_resnext50_32x4d', encoder_weights='imagenet', classes=5, activation='logsoftmax'),

#     "linknet_resnext101_8d": smp.Linknet(encoder_name='resnext101_32x8d', encoder_weights='imagenet', classes=5, activation='logsoftmax'),
#     "unet_resnext101_8d": smp.Unet(encoder_name='resnext101_32x8d', encoder_weights='imagenet', classes=5, activation='logsoftmax', decoder_attention_type='scse'), #logsoftmax ?
#  #
#     "linknet_effb4": smp.Linknet(encoder_name='efficientnet-b4', encoder_weights='imagenet', classes=5, activation='logsoftmax'),
#     "linknet_effb5": smp.Linknet(encoder_name='efficientnet-b5', encoder_weights='imagenet', classes=5, activation='logsoftmax'),
#     "linknet_effb7": smp.Linknet(encoder_name='efficientnet-b7', encoder_weights='imagenet', classes=5, activation='logsoftmax'),

#     ### EfficientNet
# }

def prediction_volume_resnet(model_path, t1_path, t1c_path, t2fl_path, outfile, mod_pred ="three_chan", tta_type=1):
    try:
        print("reading t1 ", t1_path)
        images_list = glob(t1_path)
        nii_obj_img = nib.load(images_list[0])
        image_t1 = np.copy(np.array(nii_obj_img.get_fdata()).transpose((2, 1, 0)))

        print("reading t2fl ", t2fl_path)
        images_list = glob(t2fl_path)
        nii_obj_img = nib.load(images_list[0])
        image_t2f = np.copy(np.array(nii_obj_img.get_fdata()).transpose((2, 1, 0)))

        print("reading t1c ", t1c_path)
        images_list = glob(t1c_path)
        nii_obj_img = nib.load(images_list[0])
        image_t1c = np.copy(np.array(nii_obj_img.get_fdata()).transpose((2, 1, 0)))
    except Exception as s:
        print('Failure to read all required volumes! ' + str(s))
        return str(s)
    # take template volume from old predictions to get header & affine & shape
    nii_obj_seg_template = nib.load('/src/Segmentation1-label.nii.gz')
    orig_seg = np.copy(np.array(nii_obj_seg_template.get_fdata()).transpose((2, 1, 0)))

    vol_depth, orig_width, orig_height = orig_seg.shape[0], orig_seg.shape[1], orig_seg.shape[2]
    print(orig_seg.shape)
    # vol_depth, orig_width, orig_height = image_t1c.shape[0], image_t1c.shape[1], image_t1c.shape[2]
    predicted_mask = np.zeros((vol_depth, orig_width, orig_height), dtype=int)

    from torchvision.transforms import Resize
    t_resize = Resize((orig_width, orig_height), interpolation=0)
    #print(vol_depth, orig_width, orig_height)
    with torch.set_grad_enabled(False):
        model = torch.load(model_path)
        model = model.cuda()
        model = model.eval()
        if mod_pred == 'three_chan':
            im_max = image_t1c.max()
            im_max1 = image_t1.max()
            im_max2 = image_t2f.max()
        else:
            im_max = image_t1c.max()
        for jj in range(vol_depth):
            print("jj: ", jj)
            if mod_pred == 'three_chan':

                data = image_t1c[jj]
                data = data.astype(np.float64) / im_max  # normalize the data to 0 - 1
                data = 255 * data  # Now scale by 255
                image_read = data.astype(np.uint8)
                image = np.copy(image_read)

                data = image_t1[jj]
                data = data.astype(np.float64) / im_max1  # normalize the data to 0 - 1
                data = 255 * data  # Now scale by 255
                image_read = data.astype(np.uint8)
                image1 = np.copy(image_read)

                data = image_t2f[jj]
                data = data.astype(np.float64) / im_max2  # normalize the data to 0 - 1
                data = 255 * data  # Now scale by 255
                image_read = data.astype(np.uint8)
                image2 = np.copy(image_read)

                if orig_width != 448:
                    image = cv2.resize(image, (448, 448))
                    image1 = cv2.resize(image1, (448, 448))
                    image2 = cv2.resize(image2, (448, 448))

                images_3chan = np.empty((3, image.shape[0], image.shape[1]))
                images_3chan[0, :, :] = image
                images_3chan[1, :, :] = image1
                images_3chan[2, :, :] = image2

                images_3chan = torch.FloatTensor(images_3chan[np.newaxis])

            else:

                data = image_t1c[jj]
                data = data.astype(np.float64) / im_max  # normalize the data to 0 - 1
                data = 255 * data  # Now scale by 255
                image_read = data.astype(np.uint8)
                image = np.copy(image_read)

                if orig_width != 448:
                    image = cv2.resize(image, (448, 448))

                image = image[np.newaxis]
                images_3chan = np.empty((3, image.shape[1], image.shape[2]))
                for chan_idx in range(3):
                    images_3chan[chan_idx:chan_idx + 1, :, :] = image

                images_3chan = torch.FloatTensor(images_3chan[np.newaxis])

            t_images = Variable(images_3chan.cuda())

            t_images = torch.cat([t_images, t_images.flip(-1)], 0)
            probabilities_list = []
            #print(t_images.shape)
            probabilities = model(t_images)
            probabilities = torch.stack([probabilities[0], probabilities[1].flip(-1)])
            probabilities = torch.stack([torch.mean(probabilities, 0)])
            outputs = torch.mean(probabilities, 0)
            ### need to resize to orig_w, orig_h and save all out_cut to create new NII after!!
            #print(outputs.shape)
            #outputs = outputs[0] #.data.cpu().numpy()
            #print(outputs.shape)
            predicted_mask[jj] = t_resize(outputs).argmax(axis=0).data.cpu().numpy().astype(int)
            #pred = np.zeros((outputs.shape[0], orig_width, orig_height))
            #for ii in range(outputs.shape[0]):
            #    img = outputs[ii]
            #    img = cv2.resize(img, (orig_height, orig_width), interpolation=cv2.INTER_NEAREST)
            #    pred[ii] = img

            #predicted_mask[jj] = pred.argmax(axis=0).astype(int)
    new_segmentation = nib.Nifti1Image(predicted_mask.transpose((2,1,0)), nii_obj_img.affine, nii_obj_img.header)
    nib.save(new_segmentation, outfile)
    return ""

def parse_args():
    parser = argparse.ArgumentParser(description="Brain tumor segmentation")
    parser.add_argument("--pred_type", help="prediction type: test / volume / volume_dir", default="test", type=str)
    parser.add_argument("--fold", help="fold id to train", default=0, type=int)
    parser.add_argument("--net", help="net arch", default="unet_resnet34", type=str)
    parser.add_argument("--workers", help="num workers", default=4, type=int)
    parser.add_argument("--dev", help="GPU device", default="3", type=str)
    parser.add_argument("--vol", help="volume to predict", default=" ", type=str)
    parser.add_argument("--mod_use", help="Modality used for train", default='T1c', type=str)
    parser.add_argument("--mod_pred", help="Modality to predict", default='T1c', type=str)
    parser.add_argument("--optim", help="Optimizer to use", default='adam', type=str)
    parser.add_argument("--ensemble", help="ensemble of predictions over folds", default=0, type=int)
    parser.add_argument("--brats", help="Predict BRATS", default=0, type=int)
    parser.add_argument("--dataset", help="Dataset to use for train", default='nsu', type=str)
    parser.add_argument("--tta", help="TTA type 0 -none, 1- flip, 2, 3", default=0, type=int)
    parser.add_argument("--t1_path", help="", default=0, type=str)
    parser.add_argument("--t1c_path", help="", default=0, type=str)
    parser.add_argument("--t2fl_path", help="", default=0, type=str)
    parser.add_argument("--outfile", help="", default=0, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print (f'Input args:\n {args}')
    os.environ["CUDA_VISIBLE_DEVICES"]=args.dev
    checkpoint = "/outs_multi/checkpoints_UneXt50_loss_three_chan_ralamb_full321/0.00326_UneXt50_fold0_best.pth"
    #ensemble_models, args, t1_path, t1c_path, t2_path, tta_type = 0, skullcut = False):
    # t1_path = "/tmp/autosegmentation/1.nii.gz"
    # t1c_path = "/tmp/autosegmentation/1.nii.gz"
    # t2_path = "/tmp/autosegmentation/1.nii.gz"
    # outfile = "/tmp/autosegmentation/result.nii.gz"
    prediction_volume_resnet(checkpoint, args.t1_path, args.t1c_path, args.t2fl_path, args.outfile)

