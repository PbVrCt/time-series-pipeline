import os
import pathlib

import EOD_api.EOD_api as api

TOKEN = os.environ["EOD_TOKEN"]
START = "2016-01-01"
END = "2021-02-01"
# EXCHANGE_LIST = ['LSE','US','TO','XETRA','VI','MI','BR','SW','MC','AS', 'HK','TA','KO','SG','BSE','NSE','SHE','SR','SHG','AU','SA','BA','MX']
EXCHANGE_LIST = ["US", "PA", "XETRA"]
N_STOCKS_PER_EXCHANGE = 200

# Get the first n tickers by market cap for each exchange specified
stocks = list([])
for exchange in EXCHANGE_LIST:
    stocks.extend(
        api.stock_screener(
            n_stocks=N_STOCKS_PER_EXCHANGE,
            initial_offset=0,
            token=TOKEN,
            exchange=exchange,
            maxcap=None,
            mincap=None,
        )["code"]
    )
# Download the data
df_ohlcv = api.Ohlcv(stocks, TOKEN, START, END).retrieve_data()
df_fndmt = api.Fundamental(stocks, TOKEN, START, END).retrieve_data()
# Save the data
PATH = pathlib.Path(__file__).resolve().parent / "0_raw data"
PATH.mkdir(parents=True, exist_ok=True)
df_ohlcv.to_csv(os.path.join(PATH, "raw_ohlcv.csv"))
df_fndmt.to_csv(os.path.join(PATH, "raw_fundamental.csv"))
