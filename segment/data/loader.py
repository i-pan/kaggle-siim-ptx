"""
Loaders for different datasets.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from utils.helper import channels_last_to_first, get_image_from_dicom

class RatioSampler(Sampler):
    #
    def __init__(self, data_source, num_samples, pos_neg_ratio):
        self.data_source = data_source
        self.num_samples = num_samples
        self.pos_neg_ratio = pos_neg_ratio
    #
    def __iter__(self):
        pos_indices = [i for i, _ in enumerate(self.data_source.labels) if _ == 1]
        neg_indices = [i for i, _ in enumerate(self.data_source.labels) if _ == 0]
        ratio = 1 / self.pos_neg_ratio if self.pos_neg_ratio < 1 else self.pos_neg_ratio
        if self.pos_neg_ratio > 1:
            pos_num_samples = int(self.num_samples * (ratio / (1 + ratio)))
            neg_num_samples = self.num_samples - pos_num_samples
        else:
            neg_num_samples = int(self.num_samples * (ratio / (1 + ratio)))
            pos_num_samples = self.num_samples - neg_num_samples
        pos_replace = False if len(pos_indices) >= pos_num_samples else True
        neg_replace = False if len(neg_indices) >= neg_num_samples else True
        pos = np.random.choice(pos_indices, pos_num_samples, replace=pos_replace)
        neg = np.random.choice(neg_indices, neg_num_samples, replace=neg_replace)
        combined = np.concatenate((pos, neg))
        np.random.shuffle(combined)
        return iter(list(combined))
    #
    def __len__(self):
        return self.num_samples

class XrayDataset(Dataset): 
    """
    Basic loader.
    """
    def __init__(self, imgfiles, labels, dicom=True, grayscale=True, preprocess=None, pad=None, resize=None, transform=None, tta=None, test_mode=False): 
        self.imgfiles   = imgfiles
        self.labels     = labels 
        self.dicom      = dicom
        self.grayscale  = grayscale
        self.preprocess = preprocess
        self.pad        = pad
        self.resize     = resize
        self.transform  = transform
        self.tta        = tta
        self.test_mode  = test_mode

        if transform and tta: 
            raise Exception('Cannot use both `transform` and `tta`')

    def __len__(self): 
        return len(self.imgfiles) 

    def load_image(self, imgfile):
        if self.dicom: 
            X = get_image_from_dicom(imgfile)
        else:
            if self.grayscale: 
                mode = cv2.IMREAD_GRAYSCALE
            else: 
                mode = cv2.IMREAD_COLOR
            X = cv2.imread(imgfile, mode)
        # while X is None: 
        #     i = np.random.choice(len(self.imgfiles))
        #     if self.dicom: 
        #         X = get_image_from_dicom(self.imgfiles[i])
        #     else:
        #         X = cv2.imread(self.imgfiles[i], mode)
        if self.grayscale: 
            X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        return X

    def process_image(self, X):
        if not self.test_mode:
            # Randomly flip with probability 0.5
            if np.random.binomial(1, 0.5): X = np.fliplr(X)
            # Randomly invert with probability 0.5
            if np.random.binomial(1, 0.5): X = np.invert(X)
        # 3- Apply data augmentation
        if self.transform: 
            X = self.transform(image=X)['image']
            if self.preprocess: X = self.preprocess(X)
            X = channels_last_to_first(X)
        elif self.tta: 
            X = np.asarray([ind_tta(image=X)['image'] for ind_tta in self.tta])
            if self.preprocess: X = [self.preprocess(_) for _ in X]
            X = np.asarray([channels_last_to_first(_) for _ in X])
        else:
            if self.preprocess: X = self.preprocess(X)
            X = channels_last_to_first(X)
        return X

    def __getitem__(self, i): 
        """
        Returns: x, y
            - x: tensorized input
            - y: tensorized label
        """
        X = self.load_image(self.imgfiles[i])
        # 2- Pad and resize image
        if self.pad: X = self.pad(X)
        if self.resize: X = self.resize(image=X)['image']
        X = self.process_image(X)
        y = np.asarray(self.labels[i])
        return torch.from_numpy(X.copy()).type('torch.FloatTensor'), \
               torch.from_numpy(y)


class XrayMaskDataset(Dataset): 
    """
    Basic loader.
    """
    def __init__(self, 
        imgfiles, 
        maskfiles, 
        labels, 
        multiclass=True, 
        dicom=True, 
        grayscale=True, 
        preprocess=None, 
        pad=None, 
        resize=None, 
        transform=None, 
        crop=None,
        inversion=False,
        test_mode=False): 
        self.imgfiles   = imgfiles
        self.maskfiles  = maskfiles
        self.labels     = labels 
        self.multiclass = multiclass
        self.dicom      = dicom
        self.grayscale  = grayscale
        self.preprocess = preprocess
        self.pad        = pad
        self.resize     = resize
        self.transform  = transform
        self.crop       = crop
        self.inversion  = inversion
        self.test_mode  = test_mode

    def __len__(self): 
        return len(self.imgfiles) 

    def load_image(self, imgfile):
        if self.dicom:
            X = get_image_from_dicom(imgfile)
        else:
            if self.grayscale:
                mode = cv2.IMREAD_GRAYSCALE
            else:
                mode = cv2.IMREAD_COLOR
            X = cv2.imread(imgfile, mode)
        if self.grayscale: 
            X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        return X

    def __getitem__(self, i): 
        """
        Returns: x, y
            - x: tensorized input
            - y: tensorized label
        """
        # 1- Load image
        X = self.load_image(self.imgfiles[i])
        y = cv2.imread(self.maskfiles[i], 0)
        assert np.max(y) <= 1
        if self.inversion: 
            X = 255 - X
            assert np.max(X) <= 255 and np.min(X) >= 0
        if not self.test_mode:
            # Randomly flip with probability 0.5
            if np.random.binomial(1, 0.5): 
                X = np.fliplr(X)
                y = np.fliplr(y)
            # Randomly invert with probability 0.5
            # if np.random.binomial(1, 0.5): X = np.invert(X)
        # 2- Pad and resize image
        if self.pad: 
            X = self.pad(X)
            y = self.pad(y)
        if self.resize: 
            resized = self.resize(image=X, mask=y)
            X = resized['image']
            y = resized['mask']
        # 3- Apply data augmentation
        if self.transform: 
            transformed = self.transform(image=X, mask=y)
            X = transformed['image']
            y = transformed['mask']
        # 4- Apply crop
        if self.crop:
            transformed = self.crop(image=X, mask=y)
            X = transformed['image']
            y = transformed['mask']
        # 5- Apply preprocessing
        if self.preprocess: X = self.preprocess(X)
        X = channels_last_to_first(X)
        torch_tensor_type = 'torch.FloatTensor'
        if self.labels is None:
            return torch.from_numpy(X).type(torch_tensor_type), \
                   torch.from_numpy(y).type(torch_tensor_type), \
                   torch.from_numpy(np.expand_dims(0, axis=0)).type(torch_tensor_type)                
        else:
            return torch.from_numpy(X).type(torch_tensor_type), \
                   torch.from_numpy(y).type(torch_tensor_type), \
                   torch.from_numpy(np.expand_dims(self.labels[i], axis=0)).type(torch_tensor_type)

def grid_patches(img, patch_size=512, num_rows=3, num_cols=3, return_coords=False):
    """
    Assumes image shape is (C, H, W)
    Generates <num_rows> * <num_cols> patches from an image. 
    Centers of patches gridded evenly length-/width-wise. 
    """
    if np.min(img.shape[1:]) < patch_size:
        raise Exception('Patch size {} is greater than image size {}'.format(patch_size, img.shape))
    row_start = patch_size // 2
    row_end = img.shape[1] - patch_size // 2
    col_start = patch_size // 2 
    col_end = img.shape[2] - patch_size // 2 
    row_inc = (row_end - row_start) // (num_rows - 1) 
    col_inc = (col_end - col_start) // (num_cols - 1)  
    if row_inc == 0: row_inc = 1
    if col_inc == 0: col_inc = 1  
    patch_list = [] 
    coord_list = [] 
    patch_masks = []
    for i in range(row_start, row_end+1, row_inc):
        for j in range(col_start, col_end+1, col_inc):
            patch_mask = np.zeros_like(img[0])
            x0 = i-patch_size//2 ; x1 = i+patch_size//2 
            y0 = j-patch_size//2 ; y1 = j+patch_size//2
            patch = img[:, x0:x1, y0:y1]
            patch_mask[x0:x1, y0:y1] = 1
            assert patch.shape == (img.shape[0], patch_size, patch_size)
            patch_list.append(patch)
            patch_masks.append(patch_mask)
            coord_list.append([x0,x1,y0,y1])
    if return_coords:
        return np.asarray(patch_list), coord_list
    else:
        return np.asarray(patch_list), np.asarray(patch_masks) 

class XrayCropStitchDataset(Dataset): 
    """
    Basic loader.
    """
    def __init__(self, 
        imgfiles, 
        maskfiles, 
        labels, 
        multiclass=True, 
        dicom=True, 
        grayscale=True, 
        preprocess=None, 
        pad=None, 
        resize=None, 
        transform=None, 
        crop=None,
        inversion=False,
        test_mode=False): 
        self.imgfiles   = imgfiles
        self.maskfiles  = maskfiles
        self.labels     = labels 
        self.multiclass = multiclass
        self.dicom      = dicom
        self.grayscale  = grayscale
        self.preprocess = preprocess
        self.pad        = pad
        self.resize     = resize
        self.transform  = transform
        self.crop       = crop
        self.inversion  = inversion
        self.test_mode  = test_mode

    def __len__(self): 
        return len(self.imgfiles) 

    def load_image(self, imgfile):
        if self.dicom:
            X = get_image_from_dicom(imgfile)
        else:
            if self.grayscale:
                mode = cv2.IMREAD_GRAYSCALE
            else:
                mode = cv2.IMREAD_COLOR
            X = cv2.imread(imgfile, mode)
        if self.grayscale: 
            X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        return X

    def __getitem__(self, i): 
        """
        Returns: x, y
            - x: tensorized input
            - y: tensorized label
        """
        # 1- Load image
        X = self.load_image(self.imgfiles[i])
        y = cv2.imread(self.maskfiles[i], 0)
        assert np.max(y) <= 1
        if self.inversion: 
            X = 255 - X
            assert np.max(X) <= 255 and np.min(X) >= 0
        if not self.test_mode:
            # Randomly flip with probability 0.5
            if np.random.binomial(1, 0.5): 
                X = np.fliplr(X)
                y = np.fliplr(y)
            # Randomly invert with probability 0.5
            # if np.random.binomial(1, 0.5): X = np.invert(X)
        # 2- Pad and resize image
        if self.pad: 
            X = self.pad(X)
            y = self.pad(y)
        if self.resize: 
            resized = self.resize(image=X, mask=y)
            X = resized['image']
            y = resized['mask']
        # 3- Apply data augmentation
        if self.transform: 
            transformed = self.transform(image=X, mask=y)
            X = transformed['image']
            y = transformed['mask']
        # 4- Apply crop
        if self.crop:
            transformed = self.crop(image=X, mask=y)
            X = transformed['image']
            y = transformed['mask']
        # 5- Apply preprocessing
        if self.preprocess: X = self.preprocess(X)
        X = channels_last_to_first(X)
        patches, patch_masks = grid_patches(X)
        torch_tensor_type = 'torch.FloatTensor'
        if self.labels is None:
            return torch.from_numpy(patches).type(torch_tensor_type), \
                   torch.from_numpy(patch_masks).type(torch_tensor_type), \
                   torch.from_numpy(y).type(torch_tensor_type), \
                   torch.from_numpy(np.expand_dims(0, axis=0)).type(torch_tensor_type)                
        else:
            return torch.from_numpy(patches).type(torch_tensor_type), \
                   torch.from_numpy(patch_masks).type(torch_tensor_type), \
                   torch.from_numpy(y).type(torch_tensor_type), \
                   torch.from_numpy(np.expand_dims(self.labels[i], axis=0)).type(torch_tensor_type)



