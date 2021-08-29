import pandas as pd
import numpy as np

# From Marcos Lopez de Prado's "Advances in Financial Machine Learning" chapter 3
# Creates labels that simulate stop-losses/profit-taking using the "Triple Barrier method". Example usage below


def getVol(close, span0=100, returns_timeframe=pd.Timedelta(days=1)):
    """
    Volatility estimate from returns
    Takes a dataframe with a dates index and a price column
    """
    df0 = close.index.searchsorted(close.index - returns_timeframe)
    df0 = df0[df0 > 0] - 1
    df0 = pd.Series(
        close.index[df0], index=close.index[close.shape[0] - df0.shape[0] :]
    )  # Easy to inspect for debugging
    df0 = close.loc[df0.index] / close.loc[df0].values - 1  # returns
    df0 = df0.ewm(span=span0).std()  # volatility
    return df0


def applyPtSlOnT1(close, events, PtSl):  # molecule
    """
    Returns a dataframe with timestamps of the first touches with the horizontal barriers (Profit takings and Stop losses)
    close: Dataframe (or series) of prices
    events: Dataframe with the following columns:
                    vtc: Timestamps of vertical barriers. If they are np.nan, there are no vertical barriers
                    trgt: Unit width of the horizontal barriers
                    side : Series of 1s or {1,-1}s in case the side of the bet is already decided
    PtSl:
                    PtSl[0]: Factor that multiplies the width of the top barrier. If 0 there is no top barrier
                    PtSl[1]: Factor that multiplies the width of the bottom barrier. If 0 there is no bottom barrier
    molecule: Parameter that the author used to do multipropcessing with a version of a library that is now old
    """
    # events_ = events.loc[molecule] # For multiprocessing with old version of a multiprocessing library
    events_ = (
        events  # For multiprocessing with old version of a multiprocessing library
    )
    out = events_[["vtc"]].copy(deep=True)
    # Multiply horizontal barrier widths by a factor
    if PtSl[0] > 0:
        pt = PtSl[0] * events_["trgt"]  # Profit taking
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if PtSl[1] > 0:
        sl = -PtSl[1] * events_["trgt"]  # Stop Loss
    else:
        sl = pd.Series(index=events.index)  # NaNs
    # Find the timestamps of the first barrier touches
    for loc, vtc in (
        events_["vtc"].fillna(close.index[-1]).iteritems()
    ):  # If there is no vertical barrier, it takes the last index
        df0 = close.loc[loc:vtc]
        df0 = (df0 / close[loc] - 1) * events_.at[
            loc, "side"
        ]  # Returns path # The multiplying factor is -1 in case of shorting
        out.loc[loc, "sl"] = df0[df0 < sl.loc[loc]].index.min()  # Earliest stop loss
        out.loc[loc, "pt"] = df0[
            df0 > pt.loc[loc]
        ].index.min()  # Earliest profit taking
        # print(out)
    return out


def getEvents(
    close,
    trgt,
    PtSl=[1, 1],
    minRet=0,
    vtc=False,
    vertical_length=pd.Timedelta(days=5),
    tindex=None,
    side=None,
):  # numThreads
    """
    Takes arguments and returns the ouput of applyPtSlOnT1
    close: Dataframe (or series) of prices.
    trgt: Dataframe (or series) of targets, expressed in terms of absolute returns.
    ptSl: Two non-negative floats that set the widths of the two barriers.
        If 'side' is not provided, the barriers should be symmetrical, as the bet side is not decided
    vtc: Wether to have vertical barriers
    vertical_length: Fixed length at which to create the vertical barriers
    minRet: The minimum target return required for running a triple barrier search.
    tEvents: The pandas timeindex containing the timestamps that will seed every triple barrier
    side: DataFrame with the sides of the bets, if already chosen.
    numThreads: Parameter that the author used to do multipropcessing with a version of a library that is now old
    """
    # 0) tindex is the subsample of dates to work with
    if tindex is None:
        tindex = close.index
    trgt = trgt.loc[tindex.intersection(trgt.index)]
    # 1) trgt is the width of the horizontal barriers. I pass it to getEvents as a volatility estimate of returns
    trgt = trgt[
        trgt > minRet
    ]  # Discard days whose horizontal barriers would have a small width
    trgt.columns = ["trgt"]
    # 2) PtSl_ is the horizontal widths multiplying factor
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        PtSl_ = [PtSl[0], PtSl[1]]
    else:
        side_ = side.loc[trgt.index]
        PtSl_ = [
            PtSl[0],
            PtSl[1],
        ]  # Assymetric barriers in case we have already bet ( side is not None )
    # 3) vtc is the series of vertical barriers.
    # They are set as NaT or created at a fixed length. Alternatively they could be given as an argument or be created based on horizontal width or somethign else
    if vtc is False:
        vtc = pd.Series(pd.NaT, index=tindex)
    else:
        vtc = close.index.searchsorted(close.index + vertical_length)
        vtc = vtc[vtc < close.shape[0]]
        vtc = pd.Series(
            close.index[vtc], index=close.index[: vtc.shape[0]]
        )  # NaNs at end
    # 4) Use applyPtSlOnT1
    events = pd.concat({"vtc": vtc, "trgt": trgt, "side": side_}, axis=1).dropna(
        subset=["trgt"]
    )
    # events.index = pd.to_datetime(events.index)
    df0 = applyPtSlOnT1(close=close, events=events, PtSl=PtSl_)
    # df0 = HighPerformanceComputing.MultiprocessingAndVectorization.mpPandasObj(func=applyPtSlOnT1,pdObj=("molecule", events.index),numThreads=numThreads,close=close,events=events,PtSl=PtSl_)
    # 5) Return the timestamps of the firt barrier touch, and the width that the horizontal barriers had
    # df0['sl'] = pd.to_datetime( df0['sl'] )
    # df0['pt'] = pd.to_datetime( df0['pt'] )
    events["t1"] = df0.dropna(how="all", axis=0).min(
        axis=1, skipna=True, numeric_only=False
    )
    events["t1"] = pd.to_datetime(events["t1"])
    if side is None:
        events = events.drop(["side", "vtc"], axis=1)
    return events


def getLabels(events, close):
    """
    Returns labels either 1 or -1 based on which horizontal barrier was touched first. Takes getEvents output as input
    events is a DataFrame where:
    —events.index is event's starttime
    —events[’t1’] is event's endtime
    —events[’trgt’] is event's target
    —events[’side’] (optional) implies the algo's position side
    Case 1: (’side’ not in events): label in (-1,1) <—label by price action
    Case 2: (’side’ in events): label in (0,1) <—label by pnl (meta-labeling)
    In Case 1 the only labels are {-1,1}, but the function could be modified to label cases where the vertical barrier is touched as 0s
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"]).drop_duplicates()
    px = close.reindex(px, method="bfill")
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out["t1"] = events_["t1"]
    out["ret"] = px.loc[events_["t1"]].values / px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"] *= events_["side"]
    out["Label"] = np.sign(out["ret"])  # Only labels are {-1,1}
    if "side" in events_:
        out.loc[out["ret"] <= 0, "Label"] = 0
    return out


def dropLabels(labels, minPtc=0.05):
    """
    Takes a pandas Series of Dataframe with a 'labels' column
    Drops rare labels unless there are only 2 unique label values left
    """
    while True:
        df0 = labels["Label"].value_counts(normalize=True)
        if df0.min() > minPtc or df0.shape[0] < 3:
            break
        print("Dropped Label", df0.argmin(), df0.min())
        labels = labels[labels["Label"] != df0.argmin()]
    return labels


if __name__ == "__main__":
    # Example usage for dataframe with dates index and 'Adjusted_close' column
    from io import StringIO

    df = pd.read_csv(
        StringIO(
            """
        Date    Open     High     Low   Close  Adjusted_close
    2020-10-13       1  125.390  119.65  121.10        120.7110
    2020-10-14       1  123.030  119.62  121.19        120.8008
    2020-10-15       1  121.200  118.15  120.71        120.3223
    2020-10-16       1  121.548  118.81  119.02        118.6377
    2020-10-17       1  125.390  119.65  121.10        122.7110
    2020-10-18       1  123.030  119.62  121.19        125.8008
    2020-10-19       1  121.200  118.15  120.71        121.3223
    2020-10-20       1  121.548  118.81  119.02        118.6377     
    """
        ),
        sep="\\s+",
        index_col=[0],
    )
    df.index = pd.to_datetime(df.index)
    events = getEvents(
        df["Adjusted_close"], getVol(df["Adjusted_close"], span0=100), [1, 1]
    )
    labels = getLabels(
        getEvents(
            df["Adjusted_close"], getVol(df["Adjusted_close"], span0=100), [1, 1]
        ),
        df["Adjusted_close"],
    )
    print(events)
    print(labels)

    # # Example usage for multiindex dataframe with ['Exchange','Stock','Date'] index and 'Adjusted_close' column
    # import os
    # import pathlib

    # file = "1_preprocessed_ohlcv.csv"
    # path = pathlib.Path(__file__).resolve().parent.parent / "1_processed data"
    # df = pd.read_csv(os.path.join(path, file), index_col=[0, 1])
    # df.reset_index(inplace=True)
    # df.loc[:, "Date"] = pd.to_datetime(df.loc[:, "Date"])
    # df.set_index(["Stock", "Date"], inplace=True)
    # events = df.groupby(level="Stock").apply(
    #     lambda df: getEvents(
    #         PtSl=[1, 1],
    #         close=df["Adjusted_close"].droplevel("Stock"),
    #         trgt=getVol(df["Adjusted_close"].droplevel("Stock"), span0=10),
    #     )
    # )
    # labels = df.groupby(level="Stock").apply(
    #     lambda df: getLabels(
    #         getEvents(
    #             PtSl=[1, 1],
    #             close=df["Adjusted_close"].droplevel("Stock"),
    #             trgt=getVol(df["Adjusted_close"].droplevel("Stock"), span0=10),
    #         ),
    #         df["Adjusted_close"].droplevel("Stock"),
    #     )
    # )
    # print(events)
    # print(labels)

    # Would be nice to set up a way to visualize the barriers
