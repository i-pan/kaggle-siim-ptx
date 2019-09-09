import pandas as pd
import numpy as np
import pickle
import glob
import os

from scipy.ndimage.interpolation import zoom 
from skimage.morphology import remove_small_objects
from skimage.measure import label
from tqdm import tqdm 

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;
    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 1:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;
    return " ".join(rle)

def extract_masks_from_dict(pred_dict, sop_order, real_size=1024):
    p_list = []
    t_list = []
    for sop in sop_order:
        p_list.append(pred_dict[sop]['pred_mask'])
    rescale = float(real_size) / p_list[-1].shape[-1]
    if rescale != 1:
        # Nearest neighbor
        return zoom(np.asarray(p_list), [1., 1., rescale, rescale], prefilter=False, order=0)
    else:
        return np.asarray(p_list)

def filter_small_masks(masks, min_size=3.5*1024):
    _masks = masks.copy()
    for i, m in enumerate(masks):
        if m.sum() < min_size:
            _masks[i][:] = 0
    return _masks

def remove_small_objects(masks, min_size=3.5*1024):
    _masks = masks.copy()
    for i, m in enumerate(masks):
        labels = label(m)
        m = remove_small_objects(labels, min_size)
        _masks[i][:] = (m > 0).astype('uint8')
    return _masks

# CLASSIFICATION
predictions = np.sort(glob.glob('/users/ipan/scratch/siim-ptx/segment/stage2-predictions/TRAIN_V100Flip/o0/*.csv'))
# SEGMENTATION
segmentations = np.sort(glob.glob('/users/ipan/scratch/siim-ptx/segment/stage2-predictions/TRAIN_DEEPLABXYFlip/o0/*.pkl'))
pure_segmentations = np.sort(glob.glob('/users/ipan/scratch/siim-ptx/segment/stage2-predictions/TRAIN_SEGMENTFlip/o0/*.pkl'))

with open(segmentations[0], 'rb') as f:
    pickled = pickle.load(f)

dfs = []
for pred in predictions:
    dfs.append(pd.read_csv(pred))

y_pred_mean = np.mean([_['Top0'] for _ in dfs], axis=0)
ensemble_df = pd.DataFrame({'y_pred': y_pred_mean, 'sop': dfs[0]['sop']})
sop_order = np.sort([*pickled])
y_pred_mean_dict = {sop : df['y_pred'].iloc[0] for sop, df in ensemble_df.groupby('sop')}
y_pred_mean = [y_pred_mean_dict[sop] for sop in sop_order]

seg_list = []
for seg in segmentations:
    with open(seg, 'rb') as f: 
        pickled = pickle.load(f)
    # Get predictions from classifier-segmenter dual model
    p = extract_masks_from_dict(pickled, sop_order)
    # Multiply by predictions from classifier model 
    seg_list.append(p)

for seg in pure_segmentations:
    with open(seg, 'rb') as f: 
        pickled = pickle.load(f)
    # Get predictions from classifier-segmenter dual model
    p = extract_masks_from_dict(pickled, sop_order)
    # Multiply by predictions from classifier model 
    p = np.asarray([p[i]*y_pred_mean[i] for i in range(len(p))])    
    seg_list.append(p)

dice_segmentations = np.sort(glob.glob('/users/ipan/scratch/siim-ptx/segment/lb-predictions/TRAIN_KITAMURA2/o0/*'))
dice_seg_list = []
for seg in dice_segmentations:
    with open(seg, 'rb') as f:
        pickled = pickle.load(f)
    p = extract_masks_from_dict(pickled, sop_order)
    dice_seg_list.append(p)

dice_y_seg_mean = np.mean(np.asarray(dice_seg_list), axis=0)
seg_list.append(dice_y_seg_mean)
###

def filter_small_masks(masks, min_size=3.5*1024):
    _masks = masks.copy()
    for i, m in enumerate(masks):
        if m.sum() < min_size:
            _masks[i][:] = 0
    return _masks

dscs = np.asarray([0.8579, 0.8615, 0.8647, 0.8593, 
                   0.8636, 0.8622, 0.8620, 0.8638])
dsc_cutoff = 0.857
models_to_include = [ind for ind in range(len(dscs)) if dscs[ind] > dsc_cutoff]
#models_to_include = [4,5,6,7]
dscs = dscs[models_to_include]
#weights = np.asarray([_**2 for _ in dscs])
weights = np.repeat(1., len(dscs))
#weights[-1] = 0.01
weights = weights / np.sum(weights)
__seg_list = [_ for i, _ in enumerate(seg_list) if i in models_to_include]
y_seg_mean = np.asarray([_.astype('uint8') for _ in __seg_list])
y_seg_mean = [np.average(y_seg_mean[:,ind], axis=0, weights=weights) for ind in range(y_seg_mean.shape[1])]
y_seg_mean = np.asarray(y_seg_mean)

p = (y_seg_mean > 70).astype('float32')
s = (y_seg_mean > 40).astype('float32')
#y_seg_binary = np.expand_dims(p.sum((-1,-2,-3))>0, axis=-1).astype('float32')*s
y_seg_binary = np.asarray([(p[_].sum((-1,-2)) > 0).astype('float32') * s[_] for _ in range(len(p))])
y_seg_binary = filter_small_masks(y_seg_binary, 2048.)
print('{:.1f}% PTX'.format(np.mean([1 if _.sum() > 0 else 0 for _ in y_seg_binary])*100.))

y_final_rle = [mask2rle(_.T, 1024, 1024) for _ in tqdm(y_seg_binary, total=len(y_seg_binary))]
SAVE_FILE = '../submissions/stage2-submissions/ensemble_submission002.csv'
if not os.path.exists(os.path.dirname(SAVE_FILE)): os.makedirs(os.path.dirname(SAVE_FILE))

y_final_df = pd.DataFrame({'ImageId': sop_order, 'EncodedPixels': y_final_rle})
y_final_df.loc[y_final_df['EncodedPixels'] == '', 'EncodedPixels'] = '-1'
print('{:.1f}% PTX'.format(np.mean(y_final_df['EncodedPixels'] != '-1')*100.))

y_final_df.to_csv(SAVE_FILE, index=False)

y_seg_binary = filter_small_masks(y_seg_binary, 4*1024.)
print('{:.1f}% PTX'.format(np.mean([1 if _.sum() > 0 else 0 for _ in y_seg_binary])*100.))
pos_sops = [sop_order[i] for i in range(len(y_seg_binary)) if y_seg_binary[i].sum() > 0]
y_final_df.loc[~y_final_df['ImageId'].isin(pos_sops), 'EncodedPixels'] = '-1'
print('{:.1f}% PTX'.format(np.mean(y_final_df['EncodedPixels'] != '-1')*100.))
SAVE_FILE = '../submissions/stage2-submissions/ensemble_submission006.csv'
if not os.path.exists(os.path.dirname(SAVE_FILE)): os.makedirs(os.path.dirname(SAVE_FILE))
y_final_df.to_csv(SAVE_FILE, index=False)





