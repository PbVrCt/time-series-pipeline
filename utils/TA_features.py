import pandas as pd
import numpy as np

np.seterr(divide="raise")

rolling_zscore = (
    lambda x: (x - x.rolling(window=200, min_periods=10).mean())
    / x.rolling(window=200, min_periods=10).std()
)
rolling_percentile = lambda x: x.rolling(200, min_periods=10).apply(
    lambda x: pd.Series(x).rank(pct=True)[0]
)


def PRICE_zscore(data):
    data["Adjusted_close_zscore"] = rolling_zscore(data["Adjusted_close"])
    return data


def SMA_zscore(data, ndays):
    data["SMA_zscore"] = rolling_zscore(
        data["Adjusted_close"].rolling(ndays, min_periods=5).mean()
    )
    return data


def WMA_zscore(data, ndays):
    weights = np.arange(1, ndays + 1)
    data["WMA_zscore"] = rolling_zscore(
        data["Adjusted_close"]
        .rolling(ndays, min_periods=5)
        .apply(
            lambda prices: np.dot(prices, np.arange(1, len(prices) + 1))
            / np.arange(1, len(prices) + 1).sum(),
            raw=True,
        )
    )
    return data


def EMA_zscore(data, ndays):  # emw() uses approximate formula
    data["EMA_zscore"] = rolling_zscore(
        data["Adjusted_close"].ewm(span=ndays, adjust=False, min_periods=5).mean()
    )
    return data


# Triple Exponential Moving Average
def TEMA_zscore(data, ndays):  # emv() uses approximate formula
    ema1 = data["Adjusted_close"].ewm(span=ndays, adjust=False, min_periods=5).mean()
    ema2 = ema1.ewm(span=ndays, adjust=False).mean()
    ema3 = ema2.ewm(span=ndays, adjust=False).mean()
    data["TEMA_zscore"] = rolling_zscore(((3 * ema1) - (3 * ema2) + ema3))
    return data


# Moving Average Convergence Divergence Oscilator normalized by Adjusted Close
def MACD(data, ndays, ndays2):
    data["MACD"] = (
        data["Adjusted_close"].ewm(span=ndays, adjust=False, min_periods=5).mean()
        - data["Adjusted_close"].ewm(span=ndays2, adjust=False, min_periods=5).mean()
    ) / data["Adjusted_close"]
    return data


# Commodity Channel Index / 100
def CCI(data, ndays):
    TP = (data["High"] + data["Low"] + data["Close"]) / 3
    data["CCI"] = (
        0.01
        * (TP - TP.rolling(ndays, min_periods=5).mean())
        / (0.015 * TP.rolling(ndays, min_periods=5).std())
    ).replace([np.inf, -np.inf], 0)
    return data


# Mass Index / 100
def MSSINDX(data, ndays, ndays2):
    ema1 = (
        (data["High"] - data["Low"]).ewm(span=ndays, adjust=False, min_periods=5).mean()
    )
    emaratio = ema1 / ema1.ewm(span=ndays, adjust=False, min_periods=5).mean().fillna(0)
    data["MSSINDX"] = emaratio.rolling(ndays2, min_periods=5).sum() / 100
    return data


# Aroon indicators / 100
def AROON(data, ndays):
    data["AROON_UP"] = (
        data["High"]
        .rolling(ndays + 1, min_periods=5)
        .apply(lambda x: x.argmax(), raw=True)
        / ndays
    )
    data["AROON_DOWN"] = (
        data["Low"]
        .rolling(ndays + 1, min_periods=5)
        .apply(lambda x: x.argmin(), raw=True)
        / ndays
    )
    return data


# Relative Strengh Index / 100
def RSI(data, ndays):
    delta = data["Adjusted_close"].diff(1)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.ewm(com=ndays - 1, adjust=False, min_periods=5).mean()
    roll_down = down.ewm(com=ndays - 1, adjust=False, min_periods=5).mean().abs()
    RS = roll_up / roll_down
    data["RSI"] = (1 - 1 / (1 + RS)).fillna(1)
    return data


# K (fast) stochastic oscillator / 100
def K(data, ndays):
    low = data["Low"].rolling(window=ndays, min_periods=5).min()
    high = data["High"].rolling(window=ndays, min_periods=5).max()
    data["%K"] = ((data["Close"] - low) / (high - low)).replace([np.inf, -np.inf], 0.5)
    return data


# D (slow) stochastic oscillator / 100
def D(data, ndays):
    data["%D"] = data["%K"].rolling(window=ndays).mean()
    return data


# Williams R / (-100)
def WILLSR(data, ndays):
    low = data["Low"].rolling(window=ndays, min_periods=5).min()
    high = data["High"].rolling(window=ndays, min_periods=5).max()
    data["WILLSR"] = ((high - data["Close"]) / (high - low)).replace(
        [np.inf, -np.inf], 0.5
    )
    return data


# Rate of change
def ROC(data, ndays):
    data["ROC"] = (
        data["Adjusted_close"] - data["Adjusted_close"].shift(ndays, fill_value=0)
    ) / data["Adjusted_close"].shift(
        ndays, fill_value=1000000000000000
    )  # When there is yet not enough values to calculate the ROC, set it as 0 by dividing by a big number
    return data


# Ultimate Oscillator / 100
def ULTOSC(data, ndays, ndays2, ndays3):
    trlow = np.where(
        data["Low"] > data["Close"].shift(1), data["Close"].shift(1), data["Low"]
    )
    trhigh = np.where(
        data["High"] < data["Close"].shift(1), data["Close"].shift(1), data["High"]
    )
    a = pd.DataFrame(
        data=np.transpose(np.array([data["Close"] - trlow, trhigh - trlow])),
        columns=["buypress", "trrange"],
        index=data.index,
    )
    avg = a.buypress.rolling(ndays, min_periods=5).sum() / a.trrange.rolling(
        ndays, min_periods=5
    ).sum().fillna(0.5)
    avg2 = a.buypress.rolling(ndays2, min_periods=5).sum() / a.trrange.rolling(
        ndays2, min_periods=5
    ).sum().fillna(0.5)
    avg3 = a.buypress.rolling(ndays3, min_periods=5).sum() / a.trrange.rolling(
        ndays3, min_periods=5
    ).sum().fillna(0.5)
    data["ULTOSC"] = (4 * avg + 2 * avg2 + avg3) / 7
    return data


# On Balance Volume rolling standarization
def OBV_zscore(data):
    a = np.where(
        data["Adjusted_close"] > data["Adjusted_close"].shift(1), data["Volume"], 0
    )
    b = np.where(
        data["Adjusted_close"] < data["Adjusted_close"].shift(1), -data["Volume"], 0
    )
    data["OBV_zscore"] = rolling_zscore(
        pd.DataFrame(data=np.transpose(np.array((a + b).cumsum())), index=data.index)
    )
    return data


# Volume-Price Trend but normalized by 'normdays' days volume mean
def VPT(data, normdays):
    data["VPT"] = (
        (data["Adjusted_close"] - data["Adjusted_close"].shift(1))
        * data["Volume"]
        / (
            data["Adjusted_close"].shift(1)
            * data["Volume"].rolling(window=normdays, min_periods=5).mean()
        )
    ).cumsum()
    return data


# Normalized Ease of Movement exponential moving average
def EMV(
    data, ndays, normdays
):  # ( (High-Low)/2 - (High.shift(1)-Low.shift(1))/2  ) *(High+Low)*100000000/Volume
    midpoint = (data["High"] - data["Low"]) / 2
    prev_midpoint = (data["High"].shift(1) + data["Low"].shift(1)) / 2
    midpointmove_percent = (midpoint - prev_midpoint) / midpoint
    nrmlzd_volume = (
        data["Volume"] / data["Volume"].rolling(window=normdays, min_periods=5).mean()
    )
    nrmlzd_range = (data["High"] - data["Low"]) / (data["High"] - data["Low"]).rolling(
        window=normdays, min_periods=5
    ).mean()
    data["EMV"] = (
        (midpointmove_percent * nrmlzd_range / (nrmlzd_volume * 100))
        .fillna(0)
        .rolling(ndays, min_periods=5)
        .mean()
    )
    return data


# Chaikin Oscillator but with moneyflow volume normalized by 'normdays' days volume mean
def CHKOSC(data, ndays, ndays2, normdays):
    moneyflowvol = (
        (2 * data["Close"] - data["High"] - data["Low"])
        * data["Volume"]
        / (
            (data["High"] - data["Low"])
            * data["Volume"].rolling(window=normdays, min_periods=5).mean()
        )
    )
    moneyflowvol.fillna(0, inplace=True)
    adline = moneyflowvol + moneyflowvol.shift(1)
    data["CHKOSC"] = (
        adline.ewm(span=ndays, adjust=False).mean()
        - adline.ewm(span=ndays2, adjust=False).mean()
    )
    return data


# Acumulation Distribution but with moneyflow volume normalized by 'normdays' days volume mean
def AD(data, normdays):  # TODO: Standarize in some way
    moneyflowvol = (
        (2 * data["Close"] - data["High"] - data["Low"])
        * data["Volume"]
        / (
            (data["High"] - data["Low"])
            * data["Volume"].rolling(window=normdays, min_periods=5).mean()
        )
    )
    moneyflowvol.fillna(0, inplace=True)
    data["AD"] = np.cumsum(moneyflowvol)
    return data


# Force Index Normalized by (price*volume)
def FINDX_zscore(data, ndays):
    data["FINDX_zscore"] = rolling_zscore(
        (
            ((data["Close"] - data["Close"].shift(1)) * data["Volume"])
            .fillna(0)
            .ewm(span=ndays, adjust=False, min_periods=5)
            .mean()
            / (data["Close"] * data["Volume"])
        ).replace([np.inf, -np.inf], 0)
    )
    return data


# Average True Range / Close
def ATR(data, ndays):
    trlow = np.where(
        data["Low"] > data["Close"].shift(1), data["Close"].shift(1), data["Low"]
    )
    trhigh = np.where(
        data["High"] < data["Close"].shift(1), data["Close"].shift(1), data["High"]
    )
    tr = pd.DataFrame(
        data=np.transpose(np.array([trhigh - trlow])),
        columns=["range"],
        index=data.index,
    )
    tr = pd.DataFrame(
        data=np.transpose(np.array([(trhigh - trlow) / data["Close"].to_numpy()])),
        columns=["range"],
        index=data.index,
    )
    data["ATR"] = tr.range.rolling(ndays, min_periods=5).mean()
    return data


# Chaikin volatility / 100
def CHKVLT_zscore(data, ndays):
    x = (data["High"] - data["Low"]).ewm(span=ndays, adjust=False, min_periods=5).mean()
    chkvlt = (x - x.shift(10)) / x.shift(10)
    data["CHKVLT_zscore"] = rolling_zscore(chkvlt.fillna(0))
    return data


def VOL(data, ndays):
    ret = data["Adjusted_close"] / data["Adjusted_close"].shift(1) - 1
    data["VOL"] = ret.ewm(span=ndays).std()
    return data


def add_technical_indicators(df):
    PRICE_zscore(df)
    # Trend
    SMA_zscore(df, 10)
    WMA_zscore(df, 10)
    EMA_zscore(df, 10)
    TEMA_zscore(df, 15)
    MACD(df, 12, 26)
    CCI(df, 20)
    MSSINDX(df, 9, 25)
    AROON(df, 7)
    # Momentum
    RSI(df, 14)
    K(df, 14)
    D(df, 3)
    WILLSR(df, 15)
    ROC(df, 10)
    ULTOSC(df, 7, 14, 28)
    # Volume
    OBV_zscore(df)
    # VPT(df, 60)
    EMV(df, 14, 60)
    CHKOSC(df, 3, 10, 60)
    # AD(df, 60)
    FINDX_zscore(df, 15)
    # Volatility
    # ATR(df, 14)
    CHKVLT_zscore(df, 10)
    VOL(df, 10)
    return df
