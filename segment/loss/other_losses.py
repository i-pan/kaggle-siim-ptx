from torch import nn 
import numpy as np

import torch
from torch.nn import functional as F

from .lovasz_losses import lovasz_softmax

class KLDivergence(nn.Module):
    #
    def __init__(self):
        super(KLDivergence, self).__init__()
    #
    def forward(self, N, mu, logvar):
        return 1. / N * torch.sum(mu ** 2. + logvar.exp() - 1. - logvar)

class SoftDiceLoss(nn.Module):
    #
    def __init__(self, epsilon=1e-12, per_image=True):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.per_image = per_image
    #
    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)[:,1]
        try:
            y_pred = y_pred.view(-1)
        except RuntimeError:
            y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.view(-1)
        assert y_pred.shape == y_true.shape
        if self.per_image:
            loss = 1. - (2. * torch.sum(y_true * y_pred, dim=-1) + self.epsilon) / (torch.sum(y_true ** 2., dim=-1) + torch.sum(y_pred ** 2., dim=-1) + self.epsilon)
            loss = loss.mean()
        else:
            loss = 1. - (2. * torch.sum(y_true * y_pred) + self.epsilon)/ (torch.sum(y_true ** 2.) + torch.sum(y_pred ** 2.) + self.epsilon)
        return loss

class SoftDiceLossV2(nn.Module):
    #
    def __init__(self, smooth=1., per_image=False):
        super(SoftDiceLossV2, self).__init__()
        self.smooth = smooth
        self.per_image = per_image
    #
    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)[:,1]
        try:
            y_pred = y_pred.view(-1)
        except RuntimeError:
            y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.view(-1)
        assert y_pred.shape == y_true.shape
        if self.per_image:
            loss = 1. - (2. * torch.sum(y_true * y_pred, dim=-1) + self.smooth) / (torch.sum(y_true, dim=-1) + torch.sum(y_pred, dim=-1) + self.smooth)
            loss = loss.mean()
        else:
            loss = 1. - (2. * torch.sum(y_true * y_pred) + self.smooth)/ (torch.sum(y_true) + torch.sum(y_pred) + self.smooth)
        return loss

class DiceBCELoss(nn.Module):
    # 
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    #
    def forward(self, y_prob, y_true):
        y_prob = y_prob[:,1]
        y_pred = y_prob > 0
        y_pred = y_pred.float()
        # Uses hard Dice
        dice_loss = 1. - (2. * torch.sum(y_true * y_pred) + 1)/ (torch.sum(y_true) + torch.sum(y_pred) + 1)
        bce_loss = F.binary_cross_entropy_with_logits(y_prob.flatten(), y_true.flatten())
        return (self.dice_weight * dice_loss + self.bce_weight * bce_loss) / (self.dice_weight + self.bce_weight)

class DiceBCELossV2(nn.Module):
    # 
    def __init__(self, dice_weight=0.5, bce_weight=0.5, epsilon=1e-12):
        super(DiceBCELossv2, self).__init__()
        weights_sum = dice_weight + bce_weight
        self.dice_weight = dice_weight / weights_sum
        self.bce_weight = bce_weight / weights_sum
        self.epsilon = epsilon
    #
    def forward(self, y_pred, y_true):
        y_prob = torch.softmax(y_pred, dim=1)[:,1].view(-1)
        y_pred = y_pred[:,1].view(-1)
        y_true = y_true.view(-1)
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        dice_loss = 1. - (2. * torch.sum(y_true * y_prob) + self.epsilon) / (torch.sum(y_true ** 2.) + torch.sum(y_prob ** 2.) + self.epsilon)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

class SoftDiceLovasz(nn.Module):
    #
    def __init__(self, dice_wt=0.5, lovasz_wt=0.5):
        super(SoftDiceLovasz, self).__init__()
        self.dice_wt = dice_wt 
        self.lovasz_wt = lovasz_wt
    #
    def forward(self, y_pred, y_true, only_present=False, per_image=False, ignore=None):
        y_pred = torch.softmax(y_pred, dim=1)
        dice = 1. - 2. * torch.sum(y_true * y_pred[:,1])/ (torch.sum(y_true ** 2.) + torch.sum(y_pred[:,1] ** 2.) + 1e-7)
        lovasz = lovasz_softmax(y_pred, y_true, only_present, per_image, ignore)
        return self.dice_wt * dice + self.lovasz_wt * lovasz

class BCELoss(nn.Module):
    #
    def __init__(self, pos_weight=1., neg_weight=1.):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    #
    def forward(self, y_pred, y_true, reduction='mean'):
        try:
            y_pred = y_pred[:,1].view(-1)
        except RuntimeError:
            y_pred = y_pred[:,1].contiguous().view(-1)
        y_true = y_true.view(-1)
        assert(y_pred.shape==y_true.shape)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        pos = (y_true>0.5).float()
        neg = (y_true<0.5).float()
        loss = (self.pos_weight * pos * loss) + (self.neg_weight * neg * loss)
        return loss.mean()

class WeightedBCE(nn.Module):
    # From Heng
    def __init__(self, pos_frac=0.25, neg_frac=0.75):
        super(WeightedBCE, self).__init__()
        self.pos_frac = pos_frac
        self.neg_frac = neg_frac
    #
    def forward(self, y_pred, y_true, reduction='mean'):
        try:
            y_pred = y_pred[:,1].view(-1)
        except RuntimeError:
            y_pred = y_pred[:,1].contiguous().view(-1)
        y_true = y_true.view(-1)
        assert(y_pred.shape==y_true.shape)
        #
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        #
        pos = (y_true>0.5).float()
        neg = (y_true<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.pos_frac*pos*loss/pos_weight + self.neg_frac*neg*loss/neg_weight).sum()
        #
        return loss

class WeightedBCEv2(nn.Module):
    def __init__(self):
        super(WeightedBCEv2, self).__init__()
    #
    def forward(self, y_pred, y_true, reduction='mean'):
        y_pred = y_pred[:,1].view(-1)
        y_true = y_true.view(-1)
        assert(y_pred.shape==y_true.shape)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

        pos = (y_true>0.5).float()
        neg = (y_true<0.5).float()
        pos_weight = (pos.sum().item() + 1) / len(y_true)
        neg_weight = (neg.sum().item() + 1) / len(y_true)
        pos_weight = 1 / pos_weight
        neg_weight = 1 / neg_weight 
        pos_weight = np.log(pos_weight) + 1
        neg_weight = np.log(neg_weight) + 1
        pos_weight = pos_weight / (pos_weight + neg_weight)
        neg_weight = neg_weight / (pos_weight + neg_weight)
        loss = (pos*loss*pos_weight + neg*loss*neg_weight).mean()
        return loss

# class FocalLoss(nn.Module):
#     #
#     def __init__(self, alpha=0.25, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#     #
#     def focal_loss_with_logits(y_pred, y_true):

#     def forward(self, y_pred, y_true):
#         y_pred = y_pred[:,1].view(-1)
#         y_true = y_true.view(-1)
#         assert(y_pred.shape==y_true.shape)

#         bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
#         print(bce_loss)
#         pt = torch.exp(-bce_loss)
#         print(pt)
#         focal_loss = self.alpha * (1-pt) ** self.gamma * bce_loss
#         print(focal_loss)
#         return focal_loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, y_pred, y_true):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        y_pred = y_pred[:,1].contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        if not (y_true.size() == y_pred.size()):
            raise ValueError("Target size ({}) must be the same as y_pred size ({})".format(y_true.size(), y_pred.size()))

        max_val = (-y_pred).clamp(min=0)
        loss = y_pred - y_pred * y_true + max_val + ((-max_val).exp() + (-y_pred - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-y_pred * (y_true * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=False):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()


        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
