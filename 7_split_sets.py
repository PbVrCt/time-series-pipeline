import os
import pathlib
import math

import pandas as pd
import numpy as np

from utils.split_pruning import getTrainTimes

path = pathlib.Path(__file__).resolve().parent / '1_processed data'
df = pd.read_csv( os.path.join( path , '4_merged2.csv'), index_col=[0,1], parse_dates=['Date','Label_t1'])

# Split stocks in sets
train_size = 0.6
val_size   = 0.28
def set_split( df, t, v):
    idx1 = math.ceil(  t    * df.shape[0])
    idx2 = math.ceil( (t+v) * df.shape[0])
    df.loc[:,'Set'] = None
    df.loc[:,'Set'].iloc[:idx1] = 'Train'
    df.loc[:,'Set'].iloc[idx1:idx2] = 'Validation'
    df.loc[:,'Set'].iloc[idx2:] = 'Test'
    return df
df = df.groupby( by='Stock').apply(set_split, t=train_size, v=val_size)
# 'Prune' sets whose features or labels overlap with other sets
def prune( df ):
    train =  df.loc[ df['Set']=='Train' ].droplevel('Stock')
    val   =  df.loc[ df['Set']=='Validation' ].droplevel('Stock')
    test  =  df.loc[ df['Set']=='Test' ].droplevel('Stock')
    train_timestamps = pd.Series( train['Label_t1'].values ,index=train.reset_index()['Date'].values )
    val_timestamps   = pd.Series( val['Label_t1'].values ,index=val.reset_index()['Date'].values )
    test_timestamps  = pd.Series( test['Label_t1'].values ,index=test.reset_index()['Date'].values )
    train_timestamps_pruned = pd.to_datetime( getTrainTimes( train_timestamps, val_timestamps).index, utc=True)
    val_timestamps_pruned   = pd.to_datetime( getTrainTimes( val_timestamps, test_timestamps).index, utc=True)
    test_timestamps         = pd.to_datetime(test.reset_index()['Date'].values, utc=True)
    timestamps_pruned = list(train_timestamps_pruned.union(val_timestamps_pruned).union(test_timestamps) )
    return df.droplevel('Stock').loc[timestamps_pruned]
tr_val = df.copy(deep=True) # TODO Did not prune the train+val set so the labels overlap with the test set, but this is ok for now
tr_val = tr_val.loc[ (tr_val['Set']=='Train') | (tr_val['Set']=='Validation')].drop(['Label_t1','Set'],axis=1)
df = df.groupby( by='Stock' ).apply(prune)
train =  df.loc[ df['Set']=='Train' ].drop(['Label_t1','Set'], axis=1)
val   =  df.loc[ df['Set']=='Validation' ].drop(['Label_t1','Set'], axis=1)
test  =  df.loc[ df['Set']=='Test' ].drop(['Label_t1','Set'], axis=1)
# TODO? After splitting, trim/prune sets that have instances that are standarized on rolling windows that overlap with other sets
# Balance classes by undersampling 
print('********* Before undersampling')
print( 'Train set:',train.value_counts('Label'))
print( 'Val set:',val.value_counts('Label'))
print( 'Test set:',test.value_counts('Label'))
print( 'Train and Val set:',tr_val.value_counts('Label'))
def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    nmin = df[label_col_name].value_counts().min()
    return df.groupby(label_col_name).apply(lambda x: x.sample(nmin)).droplevel(0)
train = downsample( train, 'Label')
val   = downsample( val, 'Label')
test  = downsample( test, 'Label')
print('********* After undersampling')
print( 'Train set:',train.value_counts('Label'))
print( 'Val set:',val.value_counts('Label'))
print( 'Test set:',test.value_counts('Label'))
# Sort by date in case the algos used need it later on
train.sort_index(sort_remaining=True,inplace=True)
val.sort_index(sort_remaining=True,inplace=True)
test.sort_index(sort_remaining=True,inplace=True)
tr_val.sort_index(sort_remaining=True,inplace=True)
# Split in features and labels
train_features = train.loc[:, train.columns != 'Label']
train_labels   = train.loc[:, 'Label']
val_features   = val.loc[:, val.columns != 'Label']
val_labels     = val.loc[:, 'Label']
test_features  = test.loc[:, test.columns != 'Label']
test_labels    = test.loc[:, 'Label']
tr_val_features = tr_val.loc[:, tr_val.columns != 'Label']
tr_val_labels   = tr_val.loc[:, 'Label']
# Replace -1 labels for 0s so tf.one_hot() works properly later on
train_labels.replace(-1,0, inplace=True)
val_labels.replace(-1,0, inplace=True)
test_labels.replace(-1,0, inplace=True)
tr_val_labels.replace(-1,0, inplace=True)
# Convert to numpy
train_features = train_features.to_numpy()
train_labels   = train_labels.to_numpy()
val_features   = val_features.to_numpy()
val_labels     = val_labels.to_numpy()
test_features  = test_features.to_numpy()
test_labels    = test_labels.to_numpy()
tr_val_features = tr_val_features.to_numpy()
tr_val_labels = tr_val_labels.to_numpy()

path = pathlib.Path(__file__).resolve().parent / '2_final data'
path.mkdir(parents=True, exist_ok=True)
np.save( os.path.join( path , 'train_features.npy'), train_features)
np.save( os.path.join( path , 'train_labels.npy')  , train_labels)
np.save( os.path.join( path , 'val_features.npy')  , val_features)
np.save( os.path.join( path , 'val_labels.npy')    , val_labels)
np.save( os.path.join( path , 'test_features.npy') , test_features)
np.save( os.path.join( path , 'test_labels.npy')   , test_labels)
np.save( os.path.join( path , 'train_val_features.npy'), tr_val_features)
np.save( os.path.join( path , 'train_val_labels.npy')  , tr_val_labels)
