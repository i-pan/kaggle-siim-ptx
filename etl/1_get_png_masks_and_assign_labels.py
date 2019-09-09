import pandas as pd
import numpy as np
import cv2
import os

from tqdm import tqdm

def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    #
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]
    # Need to rotate clockwise 90 degrees and flip vertically
    return np.fliplr(np.rot90(mask.reshape(width, height), 3)).astype('uint8')

train_meta = pd.read_csv('../data/train_meta.csv') 
train_rle  = pd.read_csv('../data/train-rle.csv') 

train = train_meta.merge(train_rle, left_on='sop', right_on='ImageId')

# Create binary labels for pneumothorax
train['ptx_binary'] = [0 if _ == ' -1' else 1 for _ in train[' EncodedPixels']]

TRAIN_MASKS_DIR = '../data/masks/train/'
TRAIN_MASKS_255_DIR = '../data/masks_255/train/'
if not os.path.exists(TRAIN_MASKS_DIR): os.makedirs(TRAIN_MASKS_DIR)


if not os.path.exists(TRAIN_MASKS_255_DIR): os.makedirs(TRAIN_MASKS_255_DIR)

# Generate masks from RLE and save to PNG files
# Include empty masks
mask_size_dict = {}
for pid, df in tqdm(train.groupby('ImageId'), total=len(np.unique(train['ImageId']))):
    if df[' EncodedPixels'].iloc[0] == ' -1':
        # If empty, image should only have 1 row
        # Create empty mask 
        mask = np.zeros((df['width'].iloc[0], df['height'].iloc[0])).astype('uint8')
    else:
        mask = np.zeros((df['width'].iloc[0], df['height'].iloc[0])).astype('uint8')
        for rownum, row in df.iterrows():
            mask += rle2mask(row[' EncodedPixels'], df['width'].iloc[0], df['height'].iloc[0])
    mask[mask > 1] = 1
    mask_size_dict[pid] = np.sum(mask)
    status = cv2.imwrite(os.path.join(TRAIN_MASKS_DIR, df['sop'].iloc[0] + '.png'), mask)
    mask[mask == 1] = 255
    status = cv2.imwrite(os.path.join(TRAIN_MASKS_255_DIR, df['sop'].iloc[0] + '.png'), mask)
# Mask files and image files should share same name in different folders

del train[' EncodedPixels']

train = train.drop_duplicates()
size_df = pd.DataFrame({'ImageId': list(mask_size_dict.keys()), 
                        'mask_size': [mask_size_dict[pid] for pid in mask_size_dict.keys()]})
train = train.merge(size_df, on='ImageId')

train.to_csv('../data/train_labels.csv', index=False)




