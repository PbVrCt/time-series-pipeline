import os
import pathlib

import EOD_api.EOD_api as api

token = os.environ['EOD_TOKEN']
start = "2016-01-01"
end = "2021-02-01"
# exchange_list = ['LSE','US','TO','XETRA','VI','MI','BR','SW','MC','AS', 'HK','TA','KO','SG','BSE','NSE','SHE','SR','SHG','AU','SA','BA','MX']
exchange_list = ['US','PA','XETRA']
n_stocks_per_exchange = 200

# Get the first n tickers by market cap for each exchange specified
stocks = list([])
for exchange in exchange_list:
    stocks.extend( api.stock_screener( n_stocks = n_stocks_per_exchange, initial_offset = 0,  token=token, exchange = exchange, maxcap = None, mincap= None)['code'] )
# Download the data
df_ohlcv = api.ohlcv( stocks, token, start, end).retrieve_data()
df_fndmt = api.fundamental( stocks, token, start, end).retrieve_data()

path = pathlib.Path(__file__).resolve().parent / '0_raw data'
path.mkdir(parents=True, exist_ok=True)
df_ohlcv.to_csv( os.path.join( path , 'raw_ohlcv.csv') )
df_fndmt.to_csv( os.path.join( path , 'raw_fundamental.csv') )
