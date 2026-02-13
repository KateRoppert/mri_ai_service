import numpy as np
import cv2
from albumentations import (
    HorizontalFlip,
    VerticalFlip,    
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    OneOf,
    RandomBrightnessContrast,    
    RandomGamma,
    ShiftScaleRotate
)

def strong_aug(p=0.5):
    return Compose([
        VerticalFlip(p=0.8),
        HorizontalFlip(p=0.8),              
        RandomRotate90(p=0.8),
        Transpose(p=0.8),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=10, p=0.6),
        OneOf([
            ElasticTransform(p=0.7, alpha=120, sigma=120 * 0.02, alpha_affine=120 * 0.03),
            GridDistortion(p=0.7),
            OpticalDistortion(p=0.6, distort_limit=2, shift_limit=0.2)                  
            ], p=0.8),
        RandomBrightnessContrast(p=0.8),    
        RandomGamma(p=0.7)], p=p)

def soft_aug(p=0.5):
    return Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),              
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        RandomBrightnessContrast(p=0.5),    
        RandomGamma(p=0.5)], p=p)

def softsoft_aug(p=0.5):
    return Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),              
        RandomBrightnessContrast(p=0.5),    
        RandomGamma(p=0.5)], p=p)

def affine_aug(p=0.5):
    return Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),              
        RandomRotate90(p=0.5),
        Transpose(p=0.5)], p=p)

