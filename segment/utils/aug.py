"""
Utilities for data augmentation.
"""
import cv2
import numpy as np

from albumentations import (
    Compose, OneOf, HorizontalFlip, ShiftScaleRotate, JpegCompression, Blur, CLAHE, RandomGamma, RandomContrast, RandomBrightness, Resize, PadIfNeeded, RandomCrop
)


def simple_aug(p=0.5):
    return Compose([
        #HorizontalFlip(p=0.5),
        ShiftScaleRotate(rotate_limit=10, scale_limit=0.15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=[0,0,0]),
        OneOf([
            JpegCompression(quality_lower=80),
            Blur(),
        ], p=0.5),
        OneOf([
            CLAHE(),
            RandomGamma(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.5)
    ], p=p)

def resize_aug(imsize_x, imsize_y=None):
    if imsize_y is None: imsize_y = imsize_x
    return Compose([
        Resize(imsize_x, imsize_y, always_apply=True, interpolation=cv2.INTER_CUBIC, p=1)
        ], p=1)

def crop_image(imsize_x, imsize_y=None):
    if imsize_y is None: imsize_y = imsize_x
    return Compose([
        RandomCrop(imsize_x, imsize_y, always_apply=True, p=1)
        ], p=1)

def pad_image(img, ratio=1.):
    # Default is ratio=1 aka pad to create square image
    ratio = float(ratio)
    # Given ratio, what should the height be given the width? 
    h, w = img.shape[:2]
    desired_h = int(w * ratio)
    # If the height should be greater than it is, then pad height
    if desired_h > h: 
        hdiff = int(desired_h - h) ; hdiff = int(hdiff / 2)
        pad_list = [(hdiff, desired_h-h-hdiff), (0,0), (0,0)]
    # If height should be smaller than it is, then pad width
    elif desired_h < h: 
        desired_w = int(h / ratio)
        wdiff = int(desired_w - w) ; wdiff = int(wdiff / 2)
        pad_list = [(0,0), (wdiff, desired_w-w-wdiff), (0,0)]
    elif desired_h == h: 
        return img 
    return np.pad(img, pad_list, 'constant', constant_values=np.min(img))
    

