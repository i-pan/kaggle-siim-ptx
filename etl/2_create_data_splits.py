# We will create 10 inner and 10 outer folds
# We will probably not use all of them
NSPLITS = 10

import pandas as pd 
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold

np.random.seed(88)

train_df = pd.read_csv('../data/train_labels.csv') 

# Stratify based on mask size
train_df['strata'] = 0
train_df.loc[train_df['mask_size'] > 0, 'strata'] = pd.qcut(train_df['mask_size'][train_df['mask_size'] > 0], 10, labels=range(1, 11))

train_df['outer'] = 888
outer_skf = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=88)
outer_counter = 0
for outer_train, outer_test in outer_skf.split(train_df, train_df['strata']):
    train_df.loc[outer_test, 'outer'] = outer_counter
    inner_skf = StratifiedKFold(n_splits=NSPLITS, shuffle=True, random_state=88)
    inner_counter = 0
    train_df['inner{}'.format(outer_counter)] = 888
    inner_df = train_df[train_df['outer'] != outer_counter].reset_index(drop=True)
    # Determine which IDs should be assigned to inner train
    for inner_train, inner_valid in inner_skf.split(inner_df, inner_df['strata']):
        inner_train_ids = inner_df.loc[inner_valid, 'ImageId']
        train_df.loc[train_df['ImageId'].isin(inner_train_ids), 'inner{}'.format(outer_counter)] = inner_counter
        inner_counter += 1
    outer_counter += 1

train_df.to_csv('../data/train_labels_with_splits.csv', index=False)