# Join non-standarized fundamental data on the ohlcv data for the purpose of engineering fundamental features
# that also take market data: market cap, enterprise value, price to earnings, etc. 
import os
import pathlib

import pandas as pd

path = pathlib.Path(__file__).resolve().parent / '1_processed data'
df_ohlcv = pd.read_csv( os.path.join( path , '1_preprocessed_ohlcv.csv'), index_col=[0,1], parse_dates=['Date'])
df_fndmt = pd.read_csv( os.path.join( path , '1_preprocessed_fundamental.csv'), index_col=[0,1], parse_dates=['Date'])

# Aggregate data sources of different time frecuencies. 'merge_asof' by default forward fills
df = pd.merge_asof( df_ohlcv.reset_index().sort_values(['Date']), df_fndmt.reset_index().sort_values(['Date']), on='Date',by='Stock') 
df = df.sort_values(['Stock','Date']).set_index(['Stock','Date'])

# TODO Some more automated data validation might be needed here, because for some stocks there might be a lot of ohlcv data but not fundamental data

# # Check Nans after forward filling the fundamental data on the ohlcv data
print('After the first merge (merge_asof):', df.isnull().sum().sum(), ' Nans' )
# for stock in df.index.unique(level='Stock') :
#     print( df.loc[pd.IndexSlice[stock,:],:].isnull().sum().sum(), ' Nans ', stock )

df.to_csv( os.path.join( path , '2_merged.csv') )
