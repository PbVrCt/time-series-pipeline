import os
import pathlib
import json

import numpy as np
import pandas as pd
import great_expectations as ge
from great_expectations.core.expectation_configuration import ExpectationConfiguration

import utils.TA_features as technical

path = pathlib.Path(__file__).resolve().parent / '0_raw data'
df = pd.read_csv( os.path.join( path , 'raw_ohlcv.csv'), index_col=[0,1], parse_dates=['Date'])

# Select stocks based on results of data validation
with open('preprocessing_metadata/metadata_raw_data.json') as f:
    metadata = json.loads(f.read())
    stocks_to_keep = set( metadata['Stocks'] )
df = df.copy().loc[stocks_to_keep]
#Drop duplicate index
df = df[~df.index.duplicated(keep='first')]
# Infs to Nans
print('Ohlcv')
print( df.isin([np.inf, -np.inf]).sum().sum(), ' Infs replaced to Nans')
df = df.replace([np.inf, -np.inf], np.nan)
# Forward filling
print('Before ffilling: ', df.isnull().sum().sum(), ' Nans' )
df = df.groupby(by='Stock').ffill()
print('After ffilling: ',df.isnull().sum().sum(), ' Nans')
# Feature engineering
og_columns = df.columns
df.loc[:,'Returns']      = df['Adjusted_close'].groupby(by='Stock').apply( lambda stock_df : stock_df.pct_change(1) )
# df.loc[:,'LogReturns'] = df['Adjusted_close'].groupby(by='Stock').apply( lambda stock_df : np.log( stock_df.pct_change(1) + 1 ) )
df = df.groupby(by='Stock').apply( technical.add_technical_indicators )
new_columns = set(df.columns) - set(og_columns)
# Drop some columns that don't generalize across stocks
df.drop( columns=['High', 'Low', 'Open', 'Close', 'Volume'], inplace = True )
# Trim initial dates for which the technical indicators have null values
def trim( df ):
    return df.iloc[50:].sort_index(axis=0).droplevel(0)
df = df.groupby(level='Stock').apply( trim )
# Data validation of the engineered features
min_fraction_non_nulls_per_column = 0.95
context = ge.get_context()
ge_df = ge.from_pandas(df)
for col in new_columns:
    suite = context.create_expectation_suite(expectation_suite_name='suite', overwrite_existing=True)
    expt_config = ExpectationConfiguration(expectation_type="expect_column_values_to_not_be_null",kwargs={"column": col,"mostly": min_fraction_non_nulls_per_column})
    suite.add_expectation(expectation_configuration=expt_config)
    results = ge_df.validate(expectation_suite=suite, only_return_failures=False)
    if results.success == False: 
        print(f' Results of checking for Nans for the "{col}" feature engineered: ',results)
        raise ValueError('Some of the technical indicators engineered have too many null values: '+str(col))
    if df.loc[:,col].isin([np.inf, -np.inf]).sum().sum() > 0:
        raise ValueError('Some of the technical indicators have inf values: '+str(col))
# Save names of the engineered features as not to drop them later on
with open('preprocessing_metadata/engineered_features_ohlcv.json', 'w') as f:
    json.dump( {'EngineeredFeatures': list(new_columns)}, f, indent=4)

path = pathlib.Path(__file__).resolve().parent / '1_processed data'
path.mkdir(parents=True, exist_ok=True)
df.to_csv( os.path.join( path , '1_preprocessed_ohlcv.csv') )
