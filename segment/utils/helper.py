"""
Helper functions.
"""
import pydicom
import numpy as np

import types
import math
from torch._six import inf
from collections import Counter
from functools import partial

from torch.optim.optimizer import Optimizer

def channels_last_to_first(img):
    """ Move the channels to the first dimension."""
    img = np.swapaxes(img, 0,2)
    img = np.swapaxes(img, 1,2)
    return img 

def preprocess_input(img, model): 
    """ Preprocess an input image. """
    # assume image is RGB 
    img = img[..., ::-1].astype('float32')
    model_min = model.input_range[0] ; model_max = model.input_range[1] 
    img_min = float(np.min(img)) ; img_max = float(np.max(img))
    img_range = img_max - img_min 
    model_range = model_max - model_min 
    if img_range == 0: img_range = 1. 
    img = (((img - img_min) * model_range) / img_range) + model_min 
    img[..., 0] -= model.mean[0] 
    img[..., 1] -= model.mean[1] 
    img[..., 2] -= model.mean[2] 
    img[..., 0] /= model.std[0] 
    img[..., 1] /= model.std[1] 
    img[..., 2] /= model.std[2] 
    return img

def preprocess_deeplab(img, pp_cfg): 
    """ Preprocess an input image. """
    # assume image is RGB 
    # img = img[..., ::-1].astype('float32')
    img = img.astype('float32')
    img[..., 0] -= pp_cfg['mean'][0] 
    img[..., 1] -= pp_cfg['mean'][1] 
    img[..., 2] -= pp_cfg['mean'][2] 
    img[..., 0] /= pp_cfg['std'][0] 
    img[..., 1] /= pp_cfg['std'][1] 
    img[..., 2] /= pp_cfg['std'][2] 
    return img

def preprocess_tf(img): 
    """ Preprocess an input image. """
    img = img.astype('float32')
    img /= 255. 
    img -= 0.5
    img *= 2.
    return img


def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
      num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_image_from_dicom(dicom_file): 
    """
    Extract the image as an array from a DICOM file.
    """
    dcm = pydicom.read_file(dicom_file) 
    array = dcm.pixel_array 
    try:
        array *= int(dcm.RescaleSlope)
        array += int(dcm.RescaleIntercept)
    except:
        pass 
    if dcm.PhotometricInterpretation == "MONOCHROME1": 
        array = np.invert(array.astype("uint16")) 
    array = array.astype("float32") 
    array -= np.min(array) 
    array /= np.max(array) 
    array *= 255. 
    return array.astype('uint8')

class LossTracker(): 
    #
    def __init__(self, num_moving_average=1000): 
        self.losses = []
        self.loss_history = []
        self.num_moving_average = num_moving_average
    #
    def update_loss(self, minibatch_loss): 
        self.losses.append(minibatch_loss) 
    # 
    def get_avg_loss(self): 
        self.loss_history.append(np.mean(self.losses[-self.num_moving_average:]))
        return self.loss_history[-1]
    # 
    def reset_loss(self): 
        self.losses = [] 
    # 
    def get_loss_history(self): 
        return self.loss_history

class ReduceLROnPlateau(object):


    def __init__(self, optimizer, model, classifier, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.model = model
        self.classifier = classifier
        self.best_weights = model.state_dict()
        self.best_classifier = classifier.state_dict()

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.best_weights = self.model.state_dict()
            self.best_classifier = self.classifier.state_dict()
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            print ('Restoring best weights ...')
            self.model.load_state_dict(self.best_weights)
            self.classifier.load_state_dict(self.best_classifier)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

