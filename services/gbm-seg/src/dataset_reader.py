import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import cv2

from augs import *
from albumentations import Normalize

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class CT_Dataset_png(data.Dataset):

    def __init__(self, fold=0, mode="train", datatype="binary", image_size=448, apply_aug=False,
                     normalized=False, modality='T1c', edema=True, dataset_train='nsu'):
        super(CT_Dataset_png, self).__init__()

        self.height = image_size
        self.width = image_size
        self.modality = modality
        self.edema = edema
        self.datatype = datatype
        self.mode = mode
        self.dataset_train = dataset_train

        assert mode in ("train", "valid", "test", "brats"), mode
        assert dataset_train in ("nsu", "full100", "full115", "full202", "full321", "full321cut",\
                                "fullmix", "fullmix115", "brats"), dataset_train

        if dataset_train == 'nsu':
            self.df = pd.read_csv("tables/train_val_folds.csv")
        elif dataset_train == 'brats':
            self.df = pd.read_csv("tables/BRATS_train_val_folds.csv")
        elif dataset_train == 'full100':
            self.df = pd.read_csv("tables/full100_train_val_folds.csv")
        elif dataset_train == 'full115':
            self.df = pd.read_csv("tables/full115_train_val_folds.csv")
        elif dataset_train == 'full202':
            self.df = pd.read_csv("tables/full202_train_val_folds.csv")
        elif dataset_train == 'full321':
            self.df = pd.read_csv("tables/full321_train_val_folds.csv")
        elif dataset_train == 'full321cut':
            self.df = pd.read_csv("tables/full321cut_train_val_folds.csv")
        elif dataset_train == 'fullmix':
            self.df = pd.read_csv("tables/fullmix_train_val_folds.csv")
        elif dataset_train == 'fullmix115':
            self.df = pd.read_csv("tables/fullmix115_train_val_folds.csv")

        if mode == "train":
            self.df = self.df[self.df["fold_id"] != fold]
        elif mode == "valid":
            self.df = self.df[self.df["fold_id"] == fold]
        elif mode == "test":
            if dataset_train == 'brats':
                self.df = pd.read_csv("tables/BRATS_test.csv")
            elif dataset_train == 'full100':
                self.df = pd.read_csv("tables/full100_test.csv")
            elif dataset_train == 'full115':
                self.df = pd.read_csv("tables/full115_test.csv")
            elif dataset_train == 'full202':
                self.df = pd.read_csv("tables/full202_test.csv")
            elif dataset_train == 'fullmix':
                self.df = pd.read_csv("tables/fullmix_test.csv")
            elif dataset_train == 'fullmix115':
                self.df = pd.read_csv("tables/fullmix115_test.csv")
            else:
                self.df = pd.read_csv("tables/test.csv")
        elif mode == "brats":
            self.df = pd.read_csv("tables/test_BRATS.csv")

        self.fnames = list(self.df["ImageId"])
        self.masknames = list(self.df["MaskId"])

        self.augs = False
        if mode == "train" and apply_aug == True:
            print ('Training with augmentations!')
            self.augs = True
        self.transform = softsoft_aug()  # affine_aug()  /  soft_aug()

        self.normalized = normalized
        self.norm_transform = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):

        image_id = self.fnames[index]
        mask_id = self.masknames[index]
        
        mask_read  = cv2.imread(mask_id, cv2.IMREAD_GRAYSCALE)
        width, height = mask_read.shape[0], mask_read.shape[1]

        mask = np.copy(mask_read)  # we expect to read orig values corresponding to the mask: [0,1,2,3,4]
        if self.datatype == 'binary':
            if self.edema:
                mask[np.nonzero(mask>0)] = 1.
            else:
                mask[np.nonzero(mask==4)] = 0. #assign edema as background
                mask[np.nonzero(mask>0)] = 1.

        ### allighn BRATS to NSU dataset
        if self.dataset_train in ['full100','fullmix','full115','full202','fullmix115'] and self.mode == 'brats':
            mask[np.nonzero(mask_read==2)] = 4.
            mask[np.nonzero(mask_read==4)] = 2.

        if self.modality == 'three_chan':
            image_read  = cv2.imread(image_id, cv2.IMREAD_GRAYSCALE)
            image_read1  = cv2.imread(image_id.replace('T1c_images','T1_images'), cv2.IMREAD_GRAYSCALE)
            image_read2  = cv2.imread(image_id.replace('T1c_images','T2f_images'), cv2.IMREAD_GRAYSCALE)
            if width != self.width:
                image_read = cv2.resize(image_read, (self.width, self.height))
                image_read1 = cv2.resize(image_read1, (self.width, self.height))
                image_read2 = cv2.resize(image_read2, (self.width, self.height))
                mask  = cv2.resize(mask, (self.width, self.height), interpolation = cv2.INTER_NEAREST) # to keep original mask values

            mask = mask[np.newaxis,:,:]

            images_3chan = np.empty((3,image_read.shape[0],image_read.shape[1]))
            images_3chan[0,:,:]=image_read
            images_3chan[1,:,:]=image_read1
            images_3chan[2,:,:]=image_read2

        elif self.modality == '4_chan':
            # print ('you forgot to code it!')
            image_read  = cv2.imread(image_id, cv2.IMREAD_GRAYSCALE)
            image_read1  = cv2.imread(image_id.replace('T1c_images','T1_images'), cv2.IMREAD_GRAYSCALE)
            image_read2  = cv2.imread(image_id.replace('T1c_images','T2f_images'), cv2.IMREAD_GRAYSCALE)
            image_read3  = cv2.imread(image_id.replace('T1c_images','T2_images'), cv2.IMREAD_GRAYSCALE)
            if width != self.width:
                image_read = cv2.resize(image_read, (self.width, self.height))
                image_read1 = cv2.resize(image_read1, (self.width, self.height))
                image_read2 = cv2.resize(image_read2, (self.width, self.height))
                image_read3 = cv2.resize(image_read3, (self.width, self.height))
                mask  = cv2.resize(mask, (self.width, self.height), interpolation = cv2.INTER_NEAREST) # to keep original mask values

            mask = mask[np.newaxis,:,:]

            images_3chan = np.empty((4,image_read.shape[0],image_read.shape[1]))
            images_3chan[0,:,:]=image_read
            images_3chan[1,:,:]=image_read1
            images_3chan[2,:,:]=image_read2
            images_3chan[3,:,:]=image_read3

        else:
            if self.modality == 'T1c':
                image_read  = cv2.imread(image_id, cv2.IMREAD_GRAYSCALE)
            elif self.modality == 'T1':
                image_read  = cv2.imread(image_id.replace('T1c_images','T1_images'), cv2.IMREAD_GRAYSCALE)
            elif self.modality == 'T2f':
                image_read  = cv2.imread(image_id.replace('T1c_images','T2f_images'), cv2.IMREAD_GRAYSCALE)
            elif self.modality == 'three_mix':
                image_read0  = cv2.imread(image_id, cv2.IMREAD_GRAYSCALE)/3
                image_read1  = cv2.imread(image_id.replace('T1c_images','T1_images'), cv2.IMREAD_GRAYSCALE)/3
                image_read2  = cv2.imread(image_id.replace('T1c_images','T2f_images'), cv2.IMREAD_GRAYSCALE)/3
                image_read = image_read0 + image_read1 + image_read2

            if width != self.width:
                image_read = cv2.resize(image_read, (self.width, self.height))
                mask  = cv2.resize(mask, (self.width, self.height), interpolation = cv2.INTER_NEAREST) # to keep original mask values
        
            image = np.copy(image_read)

            image = image[np.newaxis,:,:]
            mask = mask[np.newaxis,:,:]

            images_3chan = np.empty((3,image.shape[1],image.shape[2]))
            for chan_idx in range(3):
                images_3chan[chan_idx:chan_idx+1,:,:]=image

        mask = mask.transpose((2,1,0))
        images_3chan = images_3chan.astype(image_read.dtype).transpose((2,1,0))
        if self.augs:
            augmented = self.transform(image=images_3chan, mask=mask)
            images_3chan = augmented['image']
            mask = augmented['mask']

        if self.normalized:
            images_3chan = self.norm_transform(image=images_3chan)["image"]
        mask = mask.transpose((2,1,0))
        images_3chan = images_3chan.transpose((2,1,0))
        
        if self.datatype == "multi":
            return (torch.FloatTensor(images_3chan), torch.LongTensor(mask), image_id)
        else:
            return (torch.FloatTensor(images_3chan), torch.FloatTensor(mask), image_id)


class CT_Dataset_sinlge_vol(data.Dataset):

    def __init__(self, vol_path, image_size=448, normalized=False):
        super(CT_Dataset_sinlge_vol, self).__init__()

        self.height = image_size
        self.width = image_size

        images_list = glob.glob(vol_path+'/*reg_t1c.nii.gz')
        images_path = images_list[0]
        segmask_path = vol_path+'/Segmentation1-label.nii.gz'

        self.image_t1c = np.array(nib.load(images_path).get_data()).transpose((2,1,0))
        self.image_seg = np.array(nib.load(segmask_path).get_data()).transpose((2,1,0))


        self.augs = False
        if mode == "train" and apply_aug == True:
            print ('Training with augmentations!')
            self.augs = True
        self.transform = soft_aug()  # affine_aug()  /  strong_aug()

        self.normalized = normalized
        self.norm_transform = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


    def __len__(self):
        return self.image_t1c.shape[0]

    def __getitem__(self, index):

        image_read = self.image_t1c[index]
        mask_read = self.image_seg[index]
        
        width, height = image_read.shape[0], image_read.shape[1]
        if width != self.width:
            image_read = cv2.resize(image_read, (self.width, self.height))
            mask_read  = cv2.resize(mask_read, (self.width, self.height), interpolation = cv2.INTER_NEAREST) # to keep original mask values
        
        image = np.copy(image_read).astype(np.uint8)
        mask = np.copy(mask_read).astype(np.uint8)
        mask[np.nonzero(mask>0)] = 1.
        
        image = image[np.newaxis,:,:]
        mask = mask[np.newaxis,:,:]

        images_3chan = np.empty((3,image.shape[1],image.shape[2]))
        for chan_idx in range(3):
            images_3chan[chan_idx:chan_idx+1,:,:]=image

        if self.normalized:
            images_3chan = images_3chan.transpose((2,1,0))
            images_3chan = self.norm_transform(image=images_3chan)["image"]
            images_3chan = images_3chan.transpose((2,1,0))
        
        return (torch.FloatTensor(images_3chan), torch.FloatTensor(mask))




class CT_Dataset_png_multi(data.Dataset):

    def __init__(self, fold=0, mode="train", image_size=448, apply_aug=False,
                     normalized=False, modality='T1c', edema=True):
        super(CT_Dataset_png, self).__init__()

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):

        image_id = self.fnames[index]
        mask_id = self.masknames[index]
        
        mask_read  = cv2.imread(mask_id, cv2.IMREAD_GRAYSCALE)
        width, height = mask_read.shape[0], mask_read.shape[1]

        mask = np.copy(mask_read)
        if self.edema:
            mask[np.nonzero(mask>0)] = 1.
        else:
            mask[np.nonzero(mask==4)] = 0. #assign edema as background
            mask[np.nonzero(mask>0)] = 1.

        image_read  = cv2.imread(image_id, cv2.IMREAD_GRAYSCALE)
        image_read1  = cv2.imread(image_id.replace('T1c_images','T1_images'), cv2.IMREAD_GRAYSCALE)
        image_read2  = cv2.imread(image_id.replace('T1c_images','T2f_images'), cv2.IMREAD_GRAYSCALE)
        if width != self.width:
            image_read = cv2.resize(image_read, (self.width, self.height))
            image_read1 = cv2.resize(image_read1, (self.width, self.height))
            image_read2 = cv2.resize(image_read2, (self.width, self.height))
            mask  = cv2.resize(mask, (self.width, self.height), interpolation = cv2.INTER_NEAREST) # to keep original mask values

        mask = mask[np.newaxis,:,:]

        images_3chan = np.empty((3,image_read.shape[0],image_read.shape[1]))
        images_3chan[0,:,:]=image_read
        images_3chan[1,:,:]=image_read1
        images_3chan[2,:,:]=image_read2

        mask = mask.transpose((2,1,0))
        images_3chan = images_3chan.astype(image_read.dtype).transpose((2,1,0))
        if self.augs:
            augmented = self.transform(image=images_3chan, mask=mask)
            images_3chan = augmented['image']
            mask = augmented['mask']

        mask = mask.transpose((2,1,0))
        images_3chan = images_3chan.transpose((2,1,0))

        images_3chan1 = np.empty((3,images_3chan.shape[1],images_3chan.shape[2]))
        for chan_idx in range(3):
            images_3chan1[chan_idx:chan_idx+1,:,:]=images_3chan[0]

        images_3chan2 = np.empty((3,images_3chan.shape[1],images_3chan.shape[2]))
        for chan_idx in range(3):
            images_3chan2[chan_idx:chan_idx+1,:,:]=images_3chan[1]

        images_3chan3 = np.empty((3,images_3chan.shape[1],images_3chan.shape[2]))
        for chan_idx in range(3):
            images_3chan3[chan_idx:chan_idx+1,:,:]=images_3chan[2]

        
        return (torch.FloatTensor(images_3chan1), torch.FloatTensor(images_3chan2), torch.FloatTensor(images_3chan3), torch.FloatTensor(mask), image_id)


