import os
import pathlib
import json

import pandas as pd
import numpy as np

path = pathlib.Path(__file__).resolve().parent / '1_processed data'
df = pd.read_csv( os.path.join( path , '3_preprocessed2.csv'), index_col=[0,1], parse_dates=['Date'])
df_fndmt = pd.read_csv( os.path.join( path , '1_preprocessed_fundamental.csv'), index_col=[0,1], parse_dates=['Date'])

#Drop columns that were only needed for feature engineering
with open('preprocessing_metadata/metadata_raw_data.json') as f:
    metadata= json.loads(f.read())
    cols_to_drop_mixed = set( metadata['Initial fundamental features'] )
    cols_to_keep_fndmt = set( metadata['Fundamental features'] )
with open('preprocessing_metadata/engineered_features_fndmt.json') as f:
    metadata = json.loads(f.read())['EngineeredFeatures']
    cols_to_drop_mixed.union( set( metadata ) )
    cols_to_keep_fndmt.union( set( metadata ) )
df_fndmt = df_fndmt.copy().loc[:,cols_to_keep_fndmt]
df.drop( columns = list(cols_to_drop_mixed), inplace = True, errors='ignore' )
# Standarize
def zscore(x, window=10): #  TODO Add suffix _zscore to the names of the standarized columns
    r = x.rolling(window=window, min_periods=1)
    m = r.mean()
    s = r.std(ddof=0)
    masks = (s == 0) 
    z = (x-m).div(s).where( ~masks, 0) #Unintendly sets the first row to 0 instead of Nan
    return z
    # Fundamental
df_fndmt.loc[:, df_fndmt.columns != 'Period_beginning'] = df_fndmt.loc[:, df_fndmt.columns != 'Period_beginning'].groupby('Stock').apply(zscore)
df_fndmt = df_fndmt.groupby(by='Stock').apply( lambda df_fndmt: df_fndmt.droplevel(0).iloc[1:] ) # Remove the first row for each stock
    # Ohlcv and mixed
df_cols_to_standarize=['MarketCap','EnterpriseValue','EVbyRevenue','EVbyEBIT','PriceToEarnings','PriceByFreeCF','PriceToBookRatio']
df.loc[:, df_cols_to_standarize] = df.loc[:, df_cols_to_standarize].groupby('Stock').apply(zscore)
df = df.groupby(by='Stock').apply( lambda df: df.droplevel(0).iloc[1:,:] ) # Remove the first row for each stock
# Merge sources on the ohlcv data index
print('Nans in fundamental data before the second merge: ',df_fndmt.isnull().sum().sum())
print('Nans in the initially merged data before the second merge: ',df.isnull().sum().sum())
df = pd.merge_asof( df[df.columns.difference(df_fndmt.columns)].reset_index().sort_values(['Date']), df_fndmt.reset_index().sort_values(['Date']), on='Date',by='Stock').sort_values(['Stock','Date']).set_index(['Stock','Date'])
print('Nans after the second merge (merge_asof): ',df.isnull().sum().sum())

# TODO Validate data after merging sources

# Trim dates
start = "2017-01-01"
df = df.loc[pd.IndexSlice[:,start:],:]
print('After triming initial dates: ',df.isnull().sum().sum(), ' Nans')
# Replace remaining Nans for 0s
df = df.replace(np.nan, 0.0)

df.to_csv( os.path.join( path , '4_merged2.csv') )


