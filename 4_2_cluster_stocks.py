# Stock returns data is reduced from  n timestamps to m<n principal components using PCA, 
# then the stocks are clustered based on the values of the principal components using Kmeans
# Finally the clusters are saved in case that later on a different model is created for each cluster
import os
import pathlib
import json
import random

import pandas as pd

import utils.Pca_kmeans as pca

path = pathlib.Path(__file__).resolve().parent / '1_processed data'
df = pd.read_csv( os.path.join( path , '1_preprocessed_ohlcv.csv'), index_col=[0,1], parse_dates=['Date'])

# Number of clusters
n_clusters = 10
maxclusters = 3*n_clusters
# Check that the selected n_cluster and maxclusters is ok
n_stocks=len(df.index.unique(level='Stock'))
if ( n_stocks == 0 ):
    print('No stocks acepted for preprocessing.')
    exit()
elif ( n_stocks < maxclusters ) and ( n_stocks >= n_clusters ):
    maxclusters = n_stocks
elif ( n_stocks < n_clusters ):
    maxclusters = n_stocks
    n_clusters = n_stocks
# Stock returns df
df_returns = df.unstack(level='Stock')['Returns'].ffill()
df_returns.dropna(axis=0,how='any',inplace=True) # Drops stocks that lack dates after doing unstack
# PCA and K-means on the stock returns df
rnd=random.randint(0,100)
clusters = pca.pca_kmeans(df_returns, variance=0.95, transpose = True, n_cl=n_clusters, max_cl=maxclusters, seed=rnd, pcs_vs=True, expl_var=False)
print( 'Stock clusters: ', clusters )
# Save clusters
with open('preprocessing_metadata/stock_clusters.json', 'w') as f:
    json.dump( {int(k):v for k,v in clusters.items()}, f, indent=4)