import os
import pathlib
import json
import warnings

import pandas as pd
import numpy as np

from utils.make_labels import (
    getVol,
    getEvents,
    getLabels,
)  # Slow. TODO getEvents would be sped up by multiprocessing
from utils.sample_against_redundancy import (
    getIndMatrix,
    seqBootstrap,
)  # Very slow. If possible might benefit from multiprocessing/multithreading

np.seterr(divide="raise")

# Load the data
PATH = pathlib.Path(__file__).resolve().parent / "1_processed data"
df = pd.read_csv(
    os.path.join(PATH, "2_merged.csv"), index_col=[0, 1], parse_dates=["Date"]
)
# Feature engineering
og_columns = df.columns
df.loc[:, "MarketCap"] = (
    df["Adjusted_close"]
    .astype("float")
    .multiply(df["commonStockSharesOutstanding"].astype("float"))
)
df.loc[:, "EnterpriseValue"] = (
    df["MarketCap"] + df["Debt"] - df["totalCurrentAssets"].astype("float")
)  # Market Cap + Debt - CashAndCashEquivalents
df.loc[:, "EVbyRevenue"] = np.divide(
    df["EnterpriseValue"], df["totalRevenue"].astype("float")
)
df.loc[:, "EVbyEBIT"] = np.divide(df["EnterpriseValue"], df["ebit"].astype("float"))
df.loc[:, "PriceToEarnings"] = np.divide(
    df["MarketCap"], df["netIncomeApplicableToCommonShares"].astype("float")
)  # price / eps =  price /  ( (netIncome - preferred dividends) / Shares Outstanding ) = market cap / netIncomeApplicableToCommonShares
df.loc[:, "PriceByFreeCF"] = np.divide(
    df["Adjusted_close"],
    (
        df["totalCashFromOperatingActivities"].astype("float")
        - df["capitalExpenditures"].astype("float")
    ),
)  # Price / Free Cash Flow
df.loc[:, "PriceToBookRatio"] = np.divide(
    df["MarketCap"],
    (df["totalAssets"].astype("float") - df["totalLiab"].astype("float")),
)  # Price / ( BookValue / Shares Outstanding ) = Market Cap / BookValue
# Label Engineering. Labels that simulate stop-loss/profit-taking stops which are sized based on volatility
PtSl = [1, 1]  # Factors that multiply Profit-taking and Stop-loss breathds


def label(df):
    stock_close = df["Adjusted_close"].droplevel("Stock")
    events = getEvents(PtSl=PtSl, close=stock_close, trgt=getVol(stock_close, span0=10))
    return getLabels(events, stock_close)


print("Engineering labels")
labels = df.groupby(level="Stock").apply(label)
print("Done")
df = df.join(labels.loc[:, ["t1", "Label"]], on=["Stock", "Date"])
df.dropna(subset=["t1", "Label"], how="any", inplace=True)
df.rename(columns={"t1": "Label_t1"}, inplace=True)
df.drop(columns=["Adjusted_close"], inplace=True)
# Data validation. TODO More data validation on the engineered features, and also on the labels if needed
new_columns = set(df.columns) - set(og_columns)
for col in new_columns:
    if df.loc[:, col].isin([np.inf, -np.inf]).sum().sum() > 0:
        df.loc[:, col] = (
            df[col].replace([np.inf, -np.inf], np.nan).groupby(by="Stock").ffill()
        )
        warnings.warn(
            "One of the engineered features had inf values: "
            + str(col)
            + ". Infs replaced for nans and ffilled."
        )
# Save names of the engineered features as not to drop them later on
with open("preprocessing_metadata/engineered_features_mixed_sources.json", "w") as f:
    json.dump({"EngineeredFeatures": list(new_columns)}, f, indent=4)

# # Sample data in a way that attemps to reduce label redundancy. Commented because it is too slow
# def sample( df ):
#     stock_close = df['Adjusted_close'].droplevel('Stock')
#     events = getEvents( PtSl=PtSl, close= stock_close, trgt=getVol(stock_close, span0=10) )
#     barIdx = pd.date_range( start=events.reset_index()['Date'].min(), end=events['t1'].max(), freq='D')
#     timestamps = pd.Series( events['t1'].values ,index=events.reset_index()['Date'].values )
#     phi = seqBootstrap( getIndMatrix( timestamps, barIdx ) )
#     df = df.iloc()[phi].sort_index(axis=0)
#     df= df[~df.index.duplicated(keep='first')]
#     return df
# df = df.groupby(level='Stock').apply( sample )

# Save the data
df.to_csv(os.path.join(PATH, "3_preprocessed2.csv"))
