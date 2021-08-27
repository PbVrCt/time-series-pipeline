import os
import pathlib
import json
import warnings

import pandas as pd
import numpy as np

path = pathlib.Path(__file__).resolve().parent / '0_raw data'
df = pd.read_csv( os.path.join( path , 'raw_fundamental.csv'), index_col=[0,1], parse_dates=['Date','Period_beginning'])

# Select stocks and features based on results of data validation
with open('preprocessing_metadata/metadata_raw_data.json') as f:
    metadata = json.loads(f.read())
    stocks_to_keep = set( metadata['Stocks'] )
    features_to_keep = set( metadata['Fundamental features'] )
# Add some features needed for feature engineering later on
features_to_keep = features_to_keep.union(['commonStockSharesOutstanding','totalCurrentAssets','totalRevenue','netIncomeApplicableToCommonShares','totalCashFromOperatingActivities','totalAssets','totalLiab','longTermDebt','shortTermDebt','totalStockholderEquity','interestExpense'])
df = df.copy().loc[:,features_to_keep]
#Drop duplicate index
df = df[~df.index.duplicated(keep='first')]
# Drop stocks with too many missing values
df = df.copy().loc[stocks_to_keep]
# Infs to Nans
print('Fundamental')
print( df.isin([np.inf, -np.inf]).sum().sum(), ' Infs replaced to Nans')
df = df.replace([np.inf, -np.inf], np.nan)
# Forward filling
print('Before ffilling: ', df.isnull().sum().sum(), ' Nans' )
df = df.groupby(by='Stock').ffill()
print('After ffilling: ',df.isnull().sum().sum(), ' Nans')
# Feature engineering
df.loc[:,'ReturnOnEquity'] = np.divide( df['netIncome'].astype('float') , df['totalStockholderEquity'].astype('float') )
df.loc[:,'ebit']=df['netIncome'].astype('float')+df['interestExpense'].astype('float')+df['incomeTaxExpense'].astype('float')
df.loc[:,'Debt']=df['shortTermDebt'].astype('float')+df['longTermDebt'].astype('float')
# One-hot-encode categorical columns. Might not be the most appropiate way but does the job for now
if 'Period_beginning' in df.columns:
    df.loc[:,'Period_beginning'] = pd.DatetimeIndex(df['Period_beginning']).month
# Data validation of the engineered features
if df.loc[:,'ReturnOnEquity'].isin([np.inf, -np.inf]).sum().sum() > 0:
    df.loc[:,'ReturnOnEquity'] = df['ReturnOnEquity'].replace([np.inf, -np.inf], np.nan).groupby(by='Stock').ffill()
    warnings.warn('One of the fundamental features engineered had inf values: ReturnOnEquity. Infs replaced for nans and ffilled.')
# Save names of the engineered features as not to drop them later on
new_columns = ['ReturnOnEquity','ebit','Debt']
with open('preprocessing_metadata/engineered_features_fndmt.json', 'w') as f:
    json.dump( {'EngineeredFeatures': new_columns}, f, indent=4)

path = pathlib.Path(__file__).resolve().parent / '1_processed data'
path.mkdir(parents=True, exist_ok=True)
df.to_csv( os.path.join( path , '1_preprocessed_fundamental.csv') )
