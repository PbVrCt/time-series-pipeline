# Exploratory Data Analysis with Sweetviz library
import os
import pathlib

import numpy as np
import pandas as pd
import sweetviz 

path = pathlib.Path(__file__).resolve().parent / '0_raw data'
df_ohlcv = pd.read_csv(os.path.join(path, 'raw_ohlcv.csv'), index_col=[0,1], parse_dates=['Date'])
df_fndmt = pd.read_csv(os.path.join(path, 'raw_fundamental.csv'), index_col=[0,1], parse_dates=['Date','Period_beginning'])

# The library doesn't handle infs, so they have to be converted to nans
df_ohlcv = df_ohlcv.replace([np.inf, -np.inf], np.nan) 
df_fndmt = df_fndmt.replace([np.inf, -np.inf], np.nan) 
# Exploration
    # Ohlcv
my_report = sweetviz.analyze(df_ohlcv)
my_report.show_html(filepath='data exploration1_ohlcv.html')
    # Fundamental
# my_report = sweetviz.analyze(df_fndmt)
# my_report.show_html(filepath='data exploration1_fndmt.html')
# Sweetviz gives a 'Failed to allocate bitmap error' if given too many features are given at once
# In that case explore only a few features at a time:
my_report = sweetviz.analyze(df_fndmt.iloc[:,:30]) 
my_report.show_html(filepath='data exploration11_fdmt.html') 
my_report = sweetviz.analyze(df_fndmt.iloc[:,30:60]) 
my_report.show_html(filepath='data exploration12_fndmt.html') 
