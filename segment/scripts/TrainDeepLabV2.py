import sys ; sys.path.insert(0, '..') ; sys.path.insert(0, '../..')  ; sys.path.insert(0, '../../apex/')

from model.deeplab_jpu import DeepLab

from reproducibility import set_reproducibility

from data.loader import XrayMaskDataset, RatioSampler
#import loss.lovasz_losses as LL
from loss.other_losses import *

import torch
from torch import optim
from torch.optim import Optimizer
from torch import nn
import adabound

from model.train import Trainer, AllTrainer, BalancedTrainer

import argparse 
import pandas as pd 
import numpy as np 
import os 

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from utils.aug import simple_aug, resize_aug, pad_image
from utils.helper import LossTracker, preprocess_input

from torch.utils.data import DataLoader
from functools import partial 

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str)
    parser.add_argument('save_dir', type=str, help="Directory to save the trained model.") 
    parser.add_argument('data_dir', type=str, help="Directory to load image data from.") 
    parser.add_argument('mask_dir', type=str, help="Directory to load mask data from.") 
    parser.add_argument('--load-model', type=str, default='')
    parser.add_argument('--inner-fold', type=int, default=0)
    parser.add_argument('--outer-fold', type=int, default=0)
    parser.add_argument('--outer-only', action='store_true')
    parser.add_argument('--pos-only', action='store_true')
    parser.add_argument('--pos-neg-ratio', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=2000)
    parser.add_argument('--no-maxpool', action='store_true')
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--center', type=str, default='aspp')
    parser.add_argument('--jpu', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--loss', type=str, default='lovasz_softmax')
    # for WeightedBCE
    parser.add_argument('--pos-frac', type=float, default=0.25)
    parser.add_argument('--neg-frac', type=float, default=0.75)
    parser.add_argument('--grad-accum', type=float, default=0)
    parser.add_argument('--log-dampened', action='store_true')
    parser.add_argument('--labels-df', type=str, default='../../data/train_labels_with_splits.csv')
    parser.add_argument('--imsize-x', type=int, default=512)
    parser.add_argument('--imsize-y', type=int, default=512)
    parser.add_argument('--imratio', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--augment-p', type=float, default=0.5)
    parser.add_argument('--dropout-p', type=float, default=0.2)
    parser.add_argument('--steps-per-epoch', type=int, default=0)
    parser.add_argument('--train-head', type=int, default=0)
    parser.add_argument('--gn', action='store_true')
    parser.add_argument('--output-stride', type=int, default=16)
    parser.add_argument('--thresholds', type=lambda s: [float(_) for _ in s.split(',')], default=[0.1], help='Thresholds to evaluate during validation for Dice score')
    # CosineAnnealingWarmRestarts
    parser.add_argument('--cosine-anneal', action='store_true')
    parser.add_argument('--total-epochs', type=int, default=50)
    parser.add_argument('--num-snapshots', type=int, default=5)
    parser.add_argument('--eta-min', type=float, default=1e-8)
    # Optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--initial-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5) 
    # For SGD
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true')
    # For AdaBound
    parser.add_argument('--final-lr-scale', type=float, default=100.)
    parser.add_argument('--gamma', type=float, default=1e-3)

    parser.add_argument('--lr-patience', type=int, default=2) 
    parser.add_argument('--stop-patience', type=int, default=10) 
    parser.add_argument('--annealing-factor', type=float, default=0.5)
    parser.add_argument('--min-delta', type=float, default=1e-3)
    # 
    parser.add_argument('--verbosity', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=88)
    parser.add_argument('--save_best', action='store_true', help='Only store the best model.')
    
    args = parser.parse_args()
    return args 


def main():
    args = parse_args()

    set_reproducibility(args.seed)
    
    train_aug = simple_aug(p=args.augment_p)
    resize_me = resize_aug(imsize_x=args.imsize_x, imsize_y=args.imsize_y)
    pad_func  = partial(pad_image, ratio=args.imratio)
    
    print ("Training the PNEUMOTHORAX SEGMENTATION model...")
    
    torch.cuda.set_device(args.gpu) ; torch.backends.cudnn.benchmark = True 
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print("Saving model to {}".format(args.save_dir))
    print("Reading labels from {}".format(args.labels_df))
    
    df = pd.read_csv(args.labels_df)
    if args.pos_only:
        df = df[df['ptx_binary'] == 1]

    if args.outer_only:
        # We may want to only use outer splits
        train_df = df[df['outer'] != args.outer_fold]
        valid_df = df[df['outer'] == args.outer_fold]
    else:
        # Get rid of outer fold test set
        df = df[df['outer'] != args.outer_fold]
        assert np.sum(df['inner{}'.format(args.outer_fold)] == 888) == 0
        train_df = df[df['inner{}'.format(args.outer_fold)] != args.inner_fold]
        valid_df = df[df['inner{}'.format(args.outer_fold)] == args.inner_fold]
    
    print ('TRAIN: n={}'.format(len(train_df)))
    print ('% PTX: {:.1f}'.format(np.mean(train_df['ptx_binary'])*100))
    print ('VALID: n={}'.format(len(valid_df)))
    print ('% PTX: {:.1f}'.format(np.mean(valid_df['ptx_binary'])*100))
    
    print("Reading images from directory {}".format(args.data_dir))
    train_images = [os.path.join(args.data_dir, '{}.png'.format(_)) for i, _ in enumerate(train_df['sop'])]
    pos_train_images = [os.path.join(args.data_dir, '{}.png'.format(_)) for i, _ in enumerate(train_df['sop']) if train_df['ptx_binary'].iloc[i] == 1]
    neg_train_images = [os.path.join(args.data_dir, '{}.png'.format(_)) for i, _ in enumerate(train_df['sop']) if train_df['ptx_binary'].iloc[i] == 0]
    train_labels = list(train_df['ptx_binary'])

    valid_images = [os.path.join(args.data_dir, '{}.png'.format(_)) for _ in valid_df['sop']]
    valid_labels = list(valid_df['ptx_binary'])

    print("Reading masks from directory {}".format(args.mask_dir))
    train_masks = [os.path.join(args.mask_dir, '{}.png'.format(_)) for i, _ in enumerate(train_df['sop'])]
    pos_train_masks = [os.path.join(args.mask_dir, '{}.png'.format(_)) for i, _ in enumerate(train_df['sop']) if train_df['ptx_binary'].iloc[i] == 1]
    valid_masks = [os.path.join(args.mask_dir, '{}.png'.format(_)) for _ in valid_df['sop']]
    
    model = DeepLab(args.model, args.output_stride, args.gn, center=args.center, jpu=args.jpu, use_maxpool=not args.no_maxpool)
    if args.load_model != '':
        print('Loading trained model {} ...'.format(args.load_model))
        model.load_state_dict(torch.load(args.load_model))
    model = model.cuda()
    model.train()

    if args.loss == 'lovasz_softmax':
        criterion = LL.LovaszSoftmax().cuda()
    elif args.loss == 'soft_dice': 
        criterion = SoftDiceLoss().cuda()
    elif args.loss == 'soft_dicev2': 
        criterion = SoftDiceLossV2().cuda()
    elif args.loss == 'dice_bce':
        criterion = DiceBCELoss().cuda()
    elif args.loss == 'lovasz_hinge':
        criterion = LL.LovaszHinge().cuda()
    elif args.loss == 'weighted_bce':
        criterion = WeightedBCE(pos_frac=args.pos_frac, neg_frac=args.neg_frac).cuda()
    elif args.loss == 'weighted_bce_v2':
        criterion = WeightedBCEv2().cuda()
    elif args.loss == 'focal_loss':
        criterion = FocalLoss().cuda()

    train_params = model.parameters()
    
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(train_params, 
                               lr=args.initial_lr,
                               weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(train_params,
                              lr=args.initial_lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=args.nesterov)
    elif args.optimizer.lower() == 'adabound': 
        optimizer = adabound.AdaBound(train_params, 
                                      lr=args.initial_lr,
                                      final_lr=args.initial_lr * args.final_lr_scale,
                                      weight_decay=args.weight_decay,
                                      gamma=args.gamma)
    else:
        '`{}` is not a valid optimizer .'.format(args.optimizer)

    if APEX_AVAILABLE and args.mixed:
        print('Using NVIDIA Apex for mixed precision training ...')
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2", 
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )

    if not isinstance(optimizer, Optimizer):
        flag = False
        try:
            from apex.fp16_utils.fp16_optimizer import FP16_Optimizer
            if isinstance(optimizer, FP16_Optimizer):
                flag = True
        except ModuleNotFoundError:
            pass
        if not flag:
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

    if args.cosine_anneal:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=int(args.total_epochs / args.num_snapshots),
                                                                   eta_min=args.eta_min)
        scheduler.T_cur = 0.
        scheduler.mode = 'max'
        scheduler.threshold = args.min_delta
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                         factor=args.annealing_factor, 
                                                         patience=args.lr_patience, 
                                                         threshold=args.min_delta, 
                                                         threshold_mode='abs', 
                                                         verbose=True)

    # Set up preprocessing function with model 
    ppi = partial(preprocess_input, model=model)
    
    print ('Setting up data loaders ...')
    
    params = {'batch_size':  args.batch_size, 
              'shuffle':     True, 
              'num_workers': args.num_workers,
              'drop_last':   True}

    valid_params = {'batch_size':  args.batch_size, 
                    'shuffle':     False, 
                    'num_workers': args.num_workers}

    if args.balanced:
        train_set = XrayEqualMaskDataset(posfiles=pos_train_images,
                                         negfiles=neg_train_images,
                                         maskfiles=pos_train_masks,
                                         dicom=False,
                                         labels=None,
                                         preprocess=ppi, 
                                         transform=train_aug,
                                         pad=pad_func,
                                         resize=resize_me,
                                         inversion=args.invert)
    else:
        train_set = XrayMaskDataset(imgfiles=train_images,
                                    maskfiles=train_masks,
                                    dicom=False,
                                    labels=train_labels,
                                    preprocess=ppi, 
                                    transform=train_aug,
                                    pad=pad_func,
                                    resize=resize_me,
                                    inversion=args.invert)

    if args.pos_neg_ratio > 0:
        params['shuffle'] = False
        params['sampler'] = RatioSampler(train_set, args.num_samples, args.pos_neg_ratio)

    train_gen = DataLoader(train_set, **params) 
    
    valid_set = XrayMaskDataset(imgfiles=valid_images,
                                maskfiles=valid_masks,
                                dicom=False,
                                labels=valid_labels,
                                preprocess=ppi, 
                                pad=pad_func,
                                resize=resize_me,
                                test_mode=True,
                                inversion=args.invert)
    valid_gen = DataLoader(valid_set, **valid_params) 
    
    loss_tracker = LossTracker() 
    
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch == 0: 
        if args.grad_accum == 0: 
            effective_batch_size = args.batch_size
        elif args.grad_accum > 0: 
            effective_batch_size = args.batch_size * args.grad_accum
        else:
            raise Exception('`grad-accum` cannot be negative')
        if args.balanced:
            effective_batch_size *= 2
            # Hack for steps_per_epoch calculation
            train_set.imgfiles = train_set.negfiles
        steps_per_epoch = int(np.ceil(len(train_set.imgfiles) / effective_batch_size))
        if args.pos_neg_ratio > 0:
            steps_per_epoch = int(np.ceil(args.num_samples / effective_batch_size))

    if args.pos_only and args.balanced:
        raise Exception('`pos-only` and `balanced` cannot both be specified')

    trainer_class = Trainer if args.pos_only else AllTrainer
    if args.balanced:
        trainer_class = BalancedTrainer
    trainer = trainer_class(model, 'DeepLab', optimizer, criterion, loss_tracker, args.save_dir, args.save_best, multiclass=train_set.multiclass)
    #if args.pos_neg_ratio > 0:
    #    trainer.track_valid_metric = 'pos_dsc'
    trainer.grad_accum = args.grad_accum
    if APEX_AVAILABLE and args.mixed:
        trainer.use_amp = True
    trainer.set_dataloaders(train_gen, valid_gen) 
    trainer.set_thresholds(args.thresholds)

    if args.train_head > 0:
        trainer.train_head(optim.Adam(classifier.parameters()), steps_per_epoch, args.train_head)
    
    trainer.train(args.total_epochs, steps_per_epoch, scheduler, args.stop_patience, verbosity=args.verbosity)

if __name__ == '__main__':
    main()



