"""
Objects for training models.
"""

import os
import datetime
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, f1_score
from functools import partial
from loss.other_losses import KLDivergence

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from utils.helper import to_categorical

def _roc_auc_score(y_true, y_pred):
    y_true = np.asarray(y_true) 
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred[:,1])
    else:
        auc = roc_auc_score(to_categorical(y_true), y_pred, average='macro') 
    return auc

def _f1_score(y_true, y_pred):
    y_true = np.asarray(y_true) 
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) == 2:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1))
    else:
        f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='macro')     
    return f1

def _dice_score(y_true, y_pred, thresholds=[0.5]): 
    y_pred = y_pred[:,1]
    # y_pred and y_true should be (N, w, h)
    thresholds = [round(thr, 2) for thr in thresholds]
    all_dice_list = []
    all_pos_dice_list = []
    for thr in thresholds:
        dice_list = []
        pos_dice_list = []
        for i, img in enumerate(y_pred):
            img = img.copy()
            img[img >= thr] = 1
            img[img <  thr] = 0
            # Calculate Dice per image
            y_true_sum = np.sum(y_true[i])
            y_pred_sum = np.sum(img)
            if y_true_sum == 0:
                # For empty images: predicted empty mask = 1, otherwise 0
                if y_pred_sum == 0:
                    dsc = 1.
                else:
                    dsc = 0.
            else:
                dsc = f1_score(y_true[i].ravel(), img.ravel())
                pos_dice_list.append(dsc)
            dice_list.append(dsc)
        all_dice_list.append(np.mean(dice_list))
        all_pos_dice_list.append(np.mean(pos_dice_list))
    return np.max(all_dice_list), \
           thresholds[all_dice_list.index(np.max(all_dice_list))], \
           np.max(all_pos_dice_list), \
           thresholds[all_pos_dice_list.index(np.max(all_pos_dice_list))]

def _faster_dice(y_true, y_pred, thresholds=[0.5]):
    # From Heng
    size = len(y_true)
    y_pred = y_pred[:,1]
    y_pred = y_pred.reshape(size,-1)[:,::16]
    y_true = y_true.reshape(size,-1)[:,::16]
    assert np.min(y_pred) >= 0 and np.max(y_pred) <= 1
    assert(y_pred.shape == y_true.shape)

    all_dice_list = []
    all_pos_dice_list = []
    all_cls_thresholds = []
    all_seg_thresholds = []
    for cls_thres in thresholds: 
        for seg_thres in thresholds:
            p = (y_pred>cls_thres).astype('float32')
            s = (y_pred>seg_thres).astype('float32')
            #p = np.asarray([np.expand_dims((p[i].sum(-1)>0).astype('float32'), axis=-1)*s[i] for i in range(len(p))])
            p = np.expand_dims(p.sum(-1)>0, axis=-1).astype('float32')*s
            t = (y_true>0.5).astype('float32')

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = np.nonzero(t_sum==0)[0]
            pos_index = np.nonzero(t_sum>=1)[0]

            dice_neg = (p_sum==0).astype('float32')
            dice_pos = 2*(p*t).sum(-1)/((p+t).sum(-1))

            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice     = np.concatenate([dice_pos,dice_neg])

            dice_pos = np.nan_to_num(dice_pos.mean(), 0)
            dice = dice.mean()

            all_dice_list.append(dice)
            all_pos_dice_list.append(dice_pos)
            all_cls_thresholds.append(cls_thres)
            all_seg_thresholds.append(seg_thres)

    #print(pd.DataFrame({'threshold': all_thresholds, 'kaggle_dice': all_dice_list}))

    return np.max(all_dice_list), \
           all_cls_thresholds[all_dice_list.index(np.max(all_dice_list))], \
           np.max(all_pos_dice_list), \
           all_seg_thresholds[all_pos_dice_list.index(np.max(all_pos_dice_list))]

def _ultimate_kaggle_metric(y_true, y_pred, thresholds):
    # Modified from Heng
    size = len(y_true)
    y_pred = y_pred[:,1]
    y_pred = y_pred.reshape(size,-1)
    y_true = y_true.reshape(size,-1)
    assert np.min(y_pred) >= 0 and np.max(y_pred) <= 1
    assert(y_pred.shape == y_true.shape)

    all_dice_list = []
    all_pos_dice_list = []
    for cls_thres in thresholds:
        for seg_thres in thresholds:
            p = (y_pred>cls_thres).astype('float32')
            # Segmentation predictions
            s = (y_pred>seg_thres).astype('float32')

            t = (y_true>0.5).astype('float32')

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)
            neg_index = np.nonzero(t_sum==0)[0]
            pos_index = np.nonzero(t_sum>=1)[0]

            dice_neg = (p_sum==0).astype('float32')
            dice_pos = 2*(p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice     = np.concatenate([dice_pos,dice_neg])

        dice_pos = np.nan_to_num(dice_pos.mean(), 0)
        dice = dice.mean()

        all_dice_list.append(dice)
        all_pos_dice_list.append(dice_pos)

    print(pd.DataFrame({'threshold': thresholds, 'kaggle_dice': all_dice_list}))

    return np.max(all_dice_list), \
           thresholds[all_dice_list.index(np.max(all_dice_list))], \
           np.max(all_pos_dice_list), \
           thresholds[all_pos_dice_list.index(np.max(all_pos_dice_list))]

class Trainer(object): 
    def __init__(self, model, architecture, optimizer, criterion, loss_tracker, save_checkpoint, save_best, multiclass=True, validate=1):
        self.model = model 
        self.architecture = architecture 
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_tracker = loss_tracker
        self.save_checkpoint = save_checkpoint
        self.save_best = save_best
        self.best_model = None
        self.multiclass = multiclass
        self.validate_interval = validate
        self.grad_accum = 0
        self.use_amp = False
        self.track_valid_metric = 'kag'

    def set_dataloaders(self, train, valid=None): 
        self.train_gen = train 
        self.valid_gen = valid

    def set_thresholds(self, thresholds=[0.1]):
        assert type(thresholds) is list
        self.thresholds = thresholds

    def check_end_train(self): 
        return True if self.current_epoch >= self.max_epochs else False

    def train_head(self, head_optimizer, head_steps_per_epoch, head_max_epochs=5): 
        print ('Training decoder for {} epochs ...'.format(head_max_epochs))
        head_current_epoch = 0 ; head_steps = 0
        while True: 
            for i, data in enumerate(self.train_gen):
                batch, labels, class_label = data
                head_optimizer.zero_grad()
                output = self.model(batch.cuda())
                loss = self.criterion(output, labels.cuda())
                loss.backward() 
                head_optimizer.step()
                head_steps += 1
                if head_steps % head_steps_per_epoch == 0: 
                    head_current_epoch += 1
                    head_steps = 0
                    if head_current_epoch >= head_max_epochs: 
                        break
            if head_current_epoch >= head_max_epochs: 
                break
        print ('Done training decoder !')

    def save_models(self, improvement, metrics):
        cpt_name = '{arch}_{epoch}'.format(arch=self.architecture.upper(), epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))))
        for met in metrics.keys(): 
            cpt_name += '_{name}-{value:.4f}'.format(name=met.upper(), value=metrics[met])
        cpt_name += '.pth'
        if not self.save_best: 
            torch.save(self.model.state_dict(), os.path.join(self.save_checkpoint, cpt_name))
        elif improvement:
            if self.best_model is not None: 
                os.system('rm {}'.format(os.path.join(self.save_checkpoint, self.best_model)))
            self.best_model = cpt_name
            torch.save(self.model.state_dict(), os.path.join(self.save_checkpoint, cpt_name))

    def calculate_valid_metrics(self, y_true, y_pred, y_cls, loss): 
        valid_dsc, thr, pos_valid_dsc, pos_thr = _faster_dice(y_true, y_pred, thresholds=self.thresholds)
        #pos_valid_dsc, pos_thr = _dice_score_pos_only(y_true, y_pred, thresholds=self.thresholds)
        if len(np.unique(y_cls)) > 1:
            y_cls_preds = np.max(y_pred[:,1], axis=(-2, -1))
            y_cls_preds = np.repeat(np.expand_dims(y_cls_preds, axis=-1), 2, axis=-1)
            y_cls_preds[:,0] = 1. - y_cls_preds[:,0]
            valid_auc = _roc_auc_score(y_cls, y_cls_preds) 
            print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, auc = {auc:.4f}, dsc = {dsc:.4f}, thr = {thr:.2f}, dsc (pos) = {pos_dsc:.4f}, thr (pos) = {pos_thr:.2f}'
                   .format(epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))), \
                           loss=loss, \
                           auc=valid_auc, \
                           dsc=valid_dsc, \
                           thr=thr,
                           pos_dsc=pos_valid_dsc,
                           pos_thr=pos_thr))
            valid_metric = valid_dsc + valid_auc
            metrics_dict = {'auc': valid_auc, 'dsc': valid_dsc, 'thr': thr, 'pos_dsc': pos_valid_dsc, 'pos_thr': pos_thr}
        else:
            print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, dsc = {dsc:.4f}, thr = {thr:.2f}, dsc (pos) = {pos_dsc:.4f}, thr (pos) = {pos_thr:.2f}'
                   .format(epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))), \
                           loss=loss, \
                           dsc=valid_dsc, \
                           thr=thr,
                           pos_dsc=pos_valid_dsc,
                           pos_thr=pos_thr))
            valid_metric = valid_dsc
        metrics_dict = {'dsc': valid_dsc, 'thr': thr, 'pos_dsc': pos_valid_dsc, 'pos_thr': pos_thr}
        return valid_metric, metrics_dict

    def post_validate(self, valid_metric, metrics_dict):
        if self.lr_scheduler.mode == 'min': 
            improvement = valid_metric <= (self.best_valid_score - self.lr_scheduler.threshold)
        else: 
            improvement = valid_metric >= (self.best_valid_score + self.lr_scheduler.threshold) 
        if improvement: 
            self.best_valid_score = valid_metric 
            self.stopping = 0 
        else: 
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(valid_metric)
                self.stopping += 1
        self.save_models(improvement, metrics_dict)

    def validate(self): 
        with torch.no_grad():
            self.model.eval()
            self.model.cuda()
            valid_loss = 0. 
            y_pred = [] ; y_true = [] ; y_cls = []
            for i, data in tqdm(enumerate(self.valid_gen), total=len(self.valid_gen)): 
                batch, labels, class_label = data  
                output = self.model(batch.cuda())
                # Loss computed on logits
                loss = self.criterion(output, labels.cuda())
                # Softmax predictions
                y_pred.append(torch.softmax(output, dim=1).cpu().numpy())
                y_true.append(labels.numpy())
                y_cls.extend(class_label.numpy())
                valid_loss += loss.item()
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred) 
        y_cls  = np.asarray(y_cls)
        valid_loss /= float(len(self.valid_gen))
        valid_metric, metrics_dict = self.calculate_valid_metrics(y_true, y_pred, y_cls, valid_loss)
        self.post_validate(valid_metric, metrics_dict)

    def train_step(self, data): 
        batch, labels, class_label = data
        output = self.model(batch.cuda())
        if self.grad_accum > 0:
            self.loss = self.criterion(output, labels.cuda())
            self.tracker_loss += self.loss.item()
            self.grad_iter += 1
            if self.use_amp:
                with amp.scale_loss(self.loss/self.grad_accum, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                (self.loss/self.grad_accum).backward()
            if self.grad_iter % self.grad_accum == 0: 
                self.loss_tracker.update_loss(self.tracker_loss/self.grad_accum)
                self.tracker_loss = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.loss = self.criterion(output, labels.cuda())
            self.loss_tracker.update_loss(self.loss.item())
            if self.use_amp:
                with amp.scale_loss(self.loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: 
                self.loss.backward() 
            self.optimizer.step()
            self.optimizer.zero_grad()

    def train(self, max_epochs, steps_per_epoch, lr_scheduler=None, early_stopping=np.inf, freeze_bn=False, verbosity=100): 
        self.lr_scheduler = lr_scheduler 
        self.best_valid_score = 999. if lr_scheduler.mode == 'min' else 0.
        self.max_epochs = max_epochs
        self.stopping = 0
        start_time = datetime.datetime.now() ; steps = 0 
        print ('TRAINING : START')
        self.current_epoch = 0
        self.grad_iter = 0 ; self.tracker_loss = 0
        while True: 
            self.optimizer.zero_grad()
            for i, data in enumerate(self.train_gen):
                self.train_step(data)
                if self.grad_accum > 0:
                    if self.grad_iter % self.grad_accum == 0:
                        printed = False
                        steps += 1
                else:
                    printed = False    
                    steps += 1
                if isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                    self.lr_scheduler.step(self.current_epoch + steps * 1./steps_per_epoch)
                if steps % verbosity == 0 and steps > 0 and not printed:
                    duration = time.time() - step_start_time
                    duration = duration * self.grad_accum if self.grad_accum > 0 else duration
                    print('epoch {epoch}, batch {batch} / {steps_per_epoch} : loss = {train_loss:.4f} ({duration:.3f} sec/batch)'
                            .format(epoch=str(self.current_epoch + 1).zfill(len(str(max_epochs))), \
                                    batch=str(steps).zfill(len(str(steps_per_epoch))), \
                                    steps_per_epoch=steps_per_epoch, \
                                    train_loss=self.loss_tracker.get_avg_loss(), \
                                    duration=duration))
                    printed = True
                step_start_time = time.time()
                if (steps % steps_per_epoch) == 0 and steps > 0 and ((self.current_epoch + 1) % self.validate_interval) != 0:
                    steps = 0 
                elif (steps % steps_per_epoch) == 0 and steps > 0 and ((self.current_epoch + 1) % self.validate_interval) == 0: 
                    self.current_epoch += 1
                    print ('VALIDATING ...')
                    self.model.train_mode = False
                    validation_start_time = datetime.datetime.now()
                    self.validate()
                    print('Validation took {} !'.format(datetime.datetime.now() - validation_start_time))
                    steps = 0
                    # RESET BEST MODEL IF USING COSINEANNEALINGWARMRESTARTS
                    if isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                        if self.current_epoch % self.lr_scheduler.T_0 == 0:
                            self.best_model = None
                            self.best_valid_score = 999. if self.lr_scheduler.mode == 'min' else 0.
                    self.model.train()
                    self.model.cuda() 
                    self.model.train_mode = True
                    if freeze_bn:
                        for module in self.model.modules():
                            if type(module) == nn.BatchNorm2d:
                                module.eval()
                    if self.stopping >= early_stopping: 
                        # Make sure to set number of epochs to max epochs
                        self.current_epoch = max_epochs
                    if self.check_end_train():
                        # Break the for loop
                        break
            if self.check_end_train(): 
                # Break the while loop
                break 
        print ('TRAINING : END') 
        print ('Training took {}\n'.format(datetime.datetime.now() - start_time))

def turn_into_softmax(y):
    y = np.repeat(np.expand_dims(y, axis=-1), 2, axis=-1)
    y[:,0] = 1. - y[:,0]
    return y

def _kaggle_score(y_true, y_prob, average_dice=0.6, thresholds=np.linspace(0.1, 0.9, 9)):
    metric_list = []
    y_prob = y_prob[:,1]
    y_true = y_true[:,0]
    assert y_true.shape == y_prob.shape
    for thres in thresholds:
        y_pred = np.asarray([1 if _ >= thres else 0 for _ in y_prob])
        tp = np.sum(y_pred + y_true == 2)
        tn = np.sum(y_pred + y_true == 0)
        metric_list.append((tn + tp * average_dice) / len(y_true))
    return np.max(metric_list)

class AllTrainer(Trainer):
    #
    def class_score_from_pixels(self, y_cls, y_pred, top_sizes=[0, 0.1, 0.5, 1., 2.5, 5]):
        y_cls_preds = y_pred[:,1]
        y_cls_preds = y_cls_preds.reshape(y_cls_preds.shape[0], -1)
        y_cls_preds = -np.sort(-y_cls_preds, axis=1)
        auc_list = [] ; kag_list = []
        for size in top_sizes:
            tmp_size = int(y_cls_preds.shape[-1] * size / 100.) if size != 0 else 1 
            tmp_preds = np.mean(y_cls_preds[:,:tmp_size], axis=1)
            tmp_preds = turn_into_softmax(tmp_preds)
            auc_list.append(_roc_auc_score(y_cls, tmp_preds))
            kag_list.append(_kaggle_score(y_cls, tmp_preds, thresholds=self.thresholds))
        return np.max(auc_list), top_sizes[auc_list.index(np.max(auc_list))], \
               np.max(kag_list), top_sizes[kag_list.index(np.max(kag_list))]

    def calculate_valid_metrics(self, y_true, y_pred, y_cls, loss): 
        valid_dsc, thr, pos_valid_dsc, pos_thr = _faster_dice(y_true, y_pred, thresholds=self.thresholds)
        #valid_dsc, thr, pos_valid_dsc, pos_thr = _dice_score(y_true, y_pred, thresholds=self.thresholds)
        # Determine how to turn pixel scores into classification score
        _scores = self.class_score_from_pixels(y_cls, y_pred)
        valid_best_auc    = _scores[0] 
        valid_best_auctop = _scores[1]
        valid_best_kag    = _scores[2] 
        valid_best_kagtop = _scores[3]
        _toprint = 'epoch {epoch} // VALIDATION : loss = {loss:.4f}, ' + \
                   'auc = {auc:.4f}, auctop = {auctop:.4f}, ' + \
                   'kag = {kag:.4f}, kagtop = {kagtop:.4f}, ' + \
                   'dsc = {dsc:.4f}, thr = {thr:.2f}, ' + \
                   'dsc (pos) = {pos_dsc:.4f}, thr (pos) = {pos_thr:.2f}'
        print (_toprint.format(
                          epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))), \
                          loss=loss, \
                          auc=valid_best_auc, \
                          auctop=valid_best_auctop, \
                          kag=valid_best_kag, \
                          kagtop=valid_best_kagtop, \
                          dsc=valid_dsc, \
                          thr=thr, \
                          pos_dsc=pos_valid_dsc, \
                          pos_thr=pos_thr
                        )
              )
        if self.track_valid_metric == 'kag':
            valid_metric = valid_dsc
        elif self.track_valid_metric == 'pos_dsc':
            valid_metric = pos_valid_dsc
        metrics_dict = {'auc': valid_best_auc, 
                        'auctop': valid_best_auctop,
                        'kag': valid_best_kag,
                        'kagtop': valid_best_kagtop,  
                        'dsc': valid_dsc, 
                        'thr': thr, 
                        'pos_dsc': pos_valid_dsc, 
                        'pos_thr': pos_thr}
        return valid_metric, metrics_dict

class StitchTrainer(AllTrainer):
    def validate(self): 
        with torch.no_grad():
            self.model.eval()
            self.model.cuda()
            valid_loss = 0. 
            y_pred = [] ; y_true = [] ; y_cls = []
            for i, data in tqdm(enumerate(self.valid_gen), total=len(self.valid_gen)): 
                # Batch size should be 1
                patches, patch_masks, labels, class_label = data
                patches = patches[0].cuda()
                patch_masks = patch_masks[0].cuda() 
                output = self.model(patches)
                # Stitch together
                stitches = torch.zeros_like(labels).expand(output.shape[1], -1, -1).cuda() # (n, H, W)
                stitched = torch.zeros_like(labels).cuda() # (1, H, W)
                for idx, out in enumerate(output):
                    stitches[0][patch_masks[idx].long() == 1] += out[0].view(-1)
                    stitches[1][patch_masks[idx].long() == 1] += out[1].view(-1)
                    stitched[0] += patch_masks[idx]
                stitches = stitches / stitched
                # Loss computed on logits
                stitches = stitches.unsqueeze(0)
                loss = self.criterion(stitches, labels.cuda())
                # Softmax predictions
                y_pred.append(torch.softmax(stitches, dim=1).cpu().numpy())
                y_true.append(labels.numpy())
                y_cls.extend(class_label.numpy())
                valid_loss += loss.item()
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred) 
        y_cls  = np.asarray(y_cls)
        valid_loss /= float(len(self.valid_gen))
        valid_metric, metrics_dict = self.calculate_valid_metrics(y_true, y_pred, y_cls, valid_loss)
        self.post_validate(valid_metric, metrics_dict)


class BalancedTrainer(AllTrainer):
    #
    def train_step(self, data): 
        pos_batch, neg_batch, pos_labels, neg_labels, pos_class_labels, neg_class_labels = data
        batch  = torch.cat((pos_batch, neg_batch), dim=0)
        labels = torch.cat((pos_labels, neg_labels), dim=0)
        output = self.model(batch.cuda())
        if self.grad_accum > 0:
            self.loss = self.criterion(output, labels.cuda())
            self.tracker_loss += self.loss.item()
            self.grad_iter += 1
            (self.loss/self.grad_accum).backward()
            if self.grad_iter % self.grad_accum == 0: 
                self.loss_tracker.update_loss(self.tracker_loss/self.grad_accum)
                self.tracker_loss = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.loss = self.criterion(output, labels.cuda())
            self.loss_tracker.update_loss(self.loss.item()) 
            self.loss.backward() 
            self.optimizer.step()
            self.optimizer.zero_grad()


class EqualTrainer(Trainer):
    #
    def train_step(self, data): 
        pos_batch, neg_batch, pos_labels, neg_labels, pos_class_labels, neg_class_labels = data
        batch  = torch.cat((pos_batch, neg_batch), dim=0)
        labels = torch.cat((pos_labels, neg_labels), dim=0)
        class_labels = torch.cat((pos_class_labels, neg_class_labels), dim=0)
        self.optimizer.zero_grad()
        s_output = self.model(batch.cuda())
        loss = self.criterion(s_output, labels.cuda())
        self.loss_tracker.update_loss(loss.item()) 
        loss.backward() 
        self.optimizer.step()
    #
    def calculate_valid_metrics(self, y_true, y_pred, y_c_true, y_c_pred, loss):

        valid_dsc, thr = _dice_score(y_true, y_pred)
        pos_valid_dsc, pos_thr = _dice_score_pos_only(y_true, y_pred)
        print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, dsc = {dsc:.4f}, thr = {thr:.2f}, dsc (pos) = {pos_dsc:.4f}, thr (pos) = {pos_thr:.2f}'
               .format(epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))), \
                       loss=loss, \
                       dsc=valid_dsc, \
                       thr=thr,
                       pos_dsc=pos_valid_dsc,
                       pos_thr=pos_thr))
        valid_metric = pos_valid_dsc
        metrics_dict = {'dsc': valid_dsc, 'thr': thr, 'pos_dsc': pos_valid_dsc, 'pos_thr': pos_thr}
        return valid_metric, metrics_dict
    #
    def validate(self): 
        with torch.no_grad():
            self.model = self.model.eval().cuda()
            valid_loss = 0. 
            y_pred = [] ; y_true = [] ; y_c_pred = [] ; y_c_true = []
            for i, data in tqdm(enumerate(self.valid_gen), total=len(self.valid_gen)): 
                batch, labels, class_labels = data  
                s_output = self.model(batch.cuda())
                loss = self.criterion(s_output, labels.cuda()) 
                y_pred.append(s_output.cpu().numpy())
                y_true.append(labels.numpy())
                y_c_pred.append(c_output.cpu().numpy())
                y_c_true.extend(class_labels.numpy())
                valid_loss += loss.item()
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred) 
        y_c_true  = np.asarray(y_c_true)
        y_c_pred = np.vstack(y_c_pred)
        valid_loss /= float(len(self.valid_gen))
        valid_metric, metrics_dict = self.calculate_valid_metrics(y_true, y_pred, y_c_true, y_c_pred, valid_loss)
        self.post_validate(valid_metric, metrics_dict)

class EqualTrainerV2(Trainer):
    #
    def train_step(self, data): 
        pos_batch, neg_batch, pos_labels, neg_labels, pos_class_labels, neg_class_labels = data
        batch  = torch.cat((pos_batch, neg_batch), dim=0)
        labels = torch.cat((pos_labels, neg_labels), dim=0)
        class_labels = torch.cat((pos_class_labels, neg_class_labels), dim=0)
        self.optimizer.zero_grad()
        s_output, c_output = self.model(batch.cuda())
        loss = 0.5 * self.criterion[0](s_output, labels.cuda()) + 0.5 * self.criterion[1](c_output, class_labels.long().cuda())
        self.loss_tracker.update_loss(loss.item()) 
        loss.backward() 
        self.optimizer.step()
    #
    def calculate_valid_metrics(self, y_true, y_pred, y_c_true, y_c_pred, loss):
        #y_c_pred = np.repeat(np.expand_dims(y_c_pred, axis=-1), 2, axis=-1)
        #y_c_pred[:,0] = 1. - y_c_pred[:,0] 
        valid_auc = _roc_auc_score(y_c_true, y_c_pred)
        valid_dsc, thr = _dice_score(y_true, y_pred)
        pos_valid_dsc, pos_thr = _dice_score_pos_only(y_true, y_pred)
        print ('epoch {epoch} // VALIDATION : loss = {loss:.4f}, auc = {auc:.4f}, dsc = {dsc:.4f}, thr = {thr:.2f}, dsc (pos) = {pos_dsc:.4f}, thr (pos) = {pos_thr:.2f}'
               .format(epoch=str(self.current_epoch).zfill(len(str(self.max_epochs))), \
                       loss=loss, \
                       auc=valid_auc, \
                       dsc=valid_dsc, \
                       thr=thr,
                       pos_dsc=pos_valid_dsc,
                       pos_thr=pos_thr))
        valid_metric = valid_auc
        metrics_dict = {'auc': valid_auc, 'dsc': valid_dsc, 'thr': thr, 'pos_dsc': pos_valid_dsc, 'pos_thr': pos_thr}
        return valid_metric, metrics_dict
    #
    def validate(self): 
        with torch.no_grad():
            self.model = self.model.eval().cuda()
            valid_loss = 0. 
            y_pred = [] ; y_true = [] ; y_c_pred = [] ; y_c_true = []
            for i, data in tqdm(enumerate(self.valid_gen), total=len(self.valid_gen)): 
                batch, labels, class_labels = data  
                s_output, c_output = self.model(batch.cuda())
                loss = 0.5 * self.criterion[0](s_output, labels.cuda()) + 0.5 * self.criterion[1](c_output, class_labels[:,0].long().cuda())
                y_pred.append(s_output.cpu().numpy())
                y_true.append(labels.numpy())
                #y_c_pred.extend(torch.sigmoid(c_output).cpu().numpy()) #change
                y_c_pred.append(c_output.cpu().numpy())
                y_c_true.extend(class_labels.numpy())
                valid_loss += loss.item()
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred) 
        y_c_true = np.asarray(y_c_true)
        #y_c_pred = np.asarray(y_c_pred)
        y_c_pred = np.vstack(y_c_pred)
        valid_loss /= float(len(self.valid_gen))
        valid_metric, metrics_dict = self.calculate_valid_metrics(y_true, y_pred, y_c_true, y_c_pred, valid_loss)
        self.post_validate(valid_metric, metrics_dict)

kld = KLDivergence()
mse = nn.MSELoss()

class VAETrainer(Trainer):
    #
    def train_step(self, data): 
        batch, labels, class_label = data
        self.optimizer.zero_grad()
        output, recon, mu, logvar = self.model(batch.cuda())
        rescaled_output = output - torch.min(output)
        rescaled_output = rescaled_output / torch.max(rescaled_output)
        loss = self.criterion(output, labels.cuda()) + 0.1 * kld(np.prod(np.asarray(recon.shape)), mu, logvar) + 0.1 * mse(rescaled_output[:,1], recon[:,0])
        self.loss_tracker.update_loss(loss.item()) 
        loss.backward() 
        self.optimizer.step()

