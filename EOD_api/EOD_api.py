import json
import concurrent.futures
from datetime import timedelta
from datetime import datetime as dt
from abc import ABCMeta, abstractmethod
from inspect import getcallargs

import pandas as pd

pd.options.mode.chained_assignment = "raise"


def assert_arguments_in(args_to_check, allowed_values):
    def inner(f):
        def wrapper(*args, **kwargs):
            arguments = getcallargs(f, *args, **kwargs)
            for arg, values in zip(args_to_check, allowed_values):
                try:
                    assert arguments[arg] in values
                except:
                    raise ValueError(
                        "{.__name__}'s '{}' argument must be one of: {}".format(
                            f, arg, ", ".join(str(v) for v in values)
                        )
                    )
            rv = f(*args, **kwargs)
            return rv

        return wrapper

    return inner


class EodData(metaclass=ABCMeta):
    # Base class with methods common to the subclasses used to download data
    def __init__(self, tickers: list, token: str, start: str, end: str):
        self._tickers = set(tickers)
        self._token = token
        self._start = start  # String to place into the url
        self._end = end  # String to place into the url
        # The subclass' constructor/init is meant have this line: self.__df = self._download_data( self._tickers )

    @abstractmethod
    def _download_data(self, tickers: list) -> pd.DataFrame:
        pass

    def __eq__(self, comparison):
        return (
            self._tickers == comparison._tickers
            and self._token == comparison._token
            and self._start == comparison._start
            and self._end == comparison._end
            and self._df.equals(comparison._df)
        )

    def retrieve_data(self):
        # Returns the data. I formatted an index with 2 columns:
        #   A 'Date' column with dates converted to UTC using pd.to_datetime()
        #   A 'Stock' column with the tickers
        try:
            assert self._tickers != set()
        except:
            raise ValueError("Add at least 1 ticker")
        return self._df.sort_values(["Stock", "Date"]).set_index(["Stock", "Date"])

    def add_tickers(self, added_tickers):
        added_tickers = set(added_tickers) - self._tickers
        self._tickers = self._tickers.union(added_tickers)
        if added_tickers != set():
            self._df = pd.concat([self._df, self._download_data(added_tickers)])
        return self

    def remove_tickers(self, removed_tickers):
        removed_tickers = set(removed_tickers).intersection(self._tickers)
        self._tickers = self._tickers - removed_tickers
        self._df = self._df[~self._df["Stock"].isin(removed_tickers)]
        return self

    def truncate_dates(self, start, end):
        try:
            assert pd.to_datetime(start, utc=True) >= pd.to_datetime(
                self._start, utc=True
            ) and pd.to_datetime(end, utc=True) <= pd.to_datetime(self._end, utc=True)
        except:
            raise ValueError("The given dates are outside the current interval")
        self._start = pd.to_datetime(start, utc=True)
        self._end = pd.to_datetime(end, utc=True)
        self._df = (
            self._df.set_index("Date", drop=False)
            .groupby(by="Stock")
            .apply(lambda _df: _df.truncate(before=self._start, after=self._end))
            .reset_index(drop=True)
        )  # Cambiar el ultimo por inplace = True
        return self

    def _multithread_download_and_concat(self, tickers, single_thread_function):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(single_thread_function, ticker) for ticker in tickers
            ]
        futures = [f.result() for f in futures if not f.result().empty]
        if len(futures) > 1:
            df = pd.concat(futures)
        elif len(futures) == 1:
            df = futures[0]
        else:
            df = pd.DataFrame(columns=["Date", "Stock"])
        return df


class Ohlcv(EodData):
    def __init__(self, tickers, token, start, end):
        super().__init__(tickers, token, start, end)
        self._df = self._download_data(self._tickers)

    def _download_data(self, tickers):
        def historical_one_ticker(ticker):
            url = "https://eodhistoricaldata.com/api/eod/{}?from={}&to={}&api_token={}&period={}".format(
                ticker, self._start, self._end, self._token, "d"
            )
            try:
                df = pd.read_csv(
                    url,
                    usecols=[
                        "Date",
                        "Volume",
                        "Open",
                        "Close",
                        "High",
                        "Low",
                        "Adjusted_close",
                    ],
                )
            except:
                print("Failed to download ohlcv data for {}".format(ticker))
                return pd.DataFrame()
            else:
                if df.empty:
                    print("No ohlcv data for {}".format(ticker))
                    return pd.DataFrame()
                df.loc[:, "Date"] = pd.to_datetime(
                    df["Date"], errors="coerce", utc=True
                )
                df = df.copy().dropna(subset=["Date"])
                df.loc[:, "Stock"] = ticker
            return df

        df = self._multithread_download_and_concat(tickers, historical_one_ticker)
        return df


class Fundamental(EodData):
    def __init__(self, tickers, token, start, end):
        super().__init__(tickers, token, start, end)
        self._df = self._download_data(self._tickers)

    def _download_data(self, tickers):
        # As of 4/2021 the balanceSheet, cashFlow, and incmStatement from
        # 'https://eodhistoricaldata.com/api/fundamentals/{}?from={}&to={}&api_token={}&filter=Financials'
        # come with a column called 'filing_date', but if you download earnings report dates from
        # 'https://eodhistoricaldata.com/api/calendar/earnings?api_token={}&symbols={}&fmt=csv&from={}&to={}' you get a 'report_date' colummn that
        # dates 1 or a few days before the 'filing_date' column. I believe, by estimatig price volatility on those days with intraday data,
        # that the 'filing_date' is not the date the reports where realeased, but the 'report_date' is.
        # This is important for modeling and backtesting for price forecasting, so below i substitute the 'filing_date' column with the 'report_date' column.
        def earning_reports_dates(tickers):
            tickers_url = ",".join(list(tickers))
            url = "https://eodhistoricaldata.com/api/calendar/earnings?api_token={}&symbols={}&fmt=csv&from={}&to={}".format(
                self._token, tickers_url, self._start, self._end
            )
            index_df = pd.read_csv(url, usecols=["Code", "Report_Date", "Date"])
            if index_df.empty:
                # If there aren't any earning report dates in the given interval because it is too small, fetch dates starting 6 months earlier
                start_6months_earlier = str(
                    pd.to_datetime(self._start) - pd.DateOffset(months=6)
                ).split(" ")[0]
                url = "https://eodhistoricaldata.com/api/calendar/earnings?api_token={}&symbols={}&fmt=csv&from={}&to={}".format(
                    self._token, tickers_url, start_6months_earlier, self._end
                )
                index_df = pd.read_csv(url, usecols=["Code", "Report_Date", "Date"])
            index_df[["Report_Date", "Date"]] = index_df[["Report_Date", "Date"]].apply(
                pd.to_datetime, errors="coerce", utc=True, infer_datetime_format=True
            )
            index_df = index_df.copy().dropna(subset=["Report_Date", "Date"])
            index_df.rename(
                columns={
                    "Date": "Period_beginning",
                    "Report_Date": "Date",
                    "Code": "Stock",
                },
                inplace=True,
            )
            return index_df

        def fundamental_one_ticker(ticker):
            url = "https://eodhistoricaldata.com/api/fundamentals/{}?from={}&to={}&api_token={}&filter=Financials".format(
                ticker, self._start, self._end, self._token
            )
            try:
                df = pd.read_json(url).drop(["currency_symbol", "yearly"], axis=0)
                json_struct = json.loads(df.to_json(orient="split"))
                df = pd.json_normalize(json_struct)
                balanceSheet = pd.DataFrame.from_dict(df["data"][0][0][0]).T
                cashFlow = pd.DataFrame.from_dict(df["data"][0][0][1]).T
                incmStatement = pd.DataFrame.from_dict(df["data"][0][0][2]).T
                assert (
                    balanceSheet.empty == False
                    and cashFlow.empty == False
                    and incmStatement.empty == False
                )
            except:
                print("Failed download fundamental data for {}".format(ticker))
                return pd.DataFrame()
            else:
                if df.empty:
                    print("No fundamental data for {}".format(ticker))
                    return pd.DataFrame()
                df = (
                    balanceSheet.join(cashFlow, how="outer", lsuffix="_DROP")
                    .filter(regex="^(?!.*_DROP)")
                    .join(incmStatement, how="left", lsuffix="_DROP")
                    .filter(regex="^(?!.*_DROP)")
                )
                df["Stock"] = ticker
                df["date"] = pd.to_datetime(
                    df["date"], errors="coerce", utc=True, infer_datetime_format=True
                )
                df = df.copy().dropna(subset=["date"])
            return df

        index_df = earning_reports_dates(tickers)
        df = self._multithread_download_and_concat(tickers, fundamental_one_ticker)
        df = df.filter(regex="^(?!filing_date)")
        reindexed_df = index_df.merge(
            df,
            left_on=["Stock", "Period_beginning"],
            right_on=["Stock", "date"],
            how="left",
        )  # The 'Report_Date' column renamed to 'Date' in earning_report_dates() becomes the new 'Date' column for the 'reindexed_df' variable
        reindexed_df.drop("date", axis=1, inplace=True)
        return reindexed_df


class OhlcvIntraday(EodData):
    @assert_arguments_in(["intraday_frec"], [["1m", "5m"]])
    def __init__(self, tickers, token, start, end, intraday_frec):
        super().__init__(tickers, token, start, end)
        self.__frec = intraday_frec
        self._df = self._download_data(self._tickers)

    def _download_data(self, tickers):
        def intraday_one_ticker(ticker):
            def intraday_one_ticker_100_days(start, end):
                start = str(start.timestamp())
                end = str(end.timestamp())
                url = "https://eodhistoricaldata.com/api/intraday/{}?api_token={}&fmt=csv&from={}&to={}&interval={}".format(
                    ticker, token, start, end, self.__frec
                )
                try:
                    df = pd.read_csv(
                        url,
                        usecols=[
                            "Timestamp",
                            "Gmtoffset",
                            "Datetime",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Volume",
                        ],
                    )  # Gmtoffset comes in seconds, but as of 4/2021 comes only with value 0
                except:
                    print(
                        "Failed to download intraday data for {} betwen {} and {}".format(
                            ticker,
                            dt.fromtimestamp(int(float(start))),
                            dt.fromtimestamp(int(float(end))),
                        )
                    )
                    return pd.DataFrame()
                else:
                    if df.empty:
                        print(
                            "No intraday data for {} betwen {} and {}".format(
                                ticker,
                                dt.fromtimestamp(int(float(start))),
                                dt.fromtimestamp(int(float(end))),
                            )
                        )
                        return pd.DataFrame()
                return df

            amount_days = (pd.to_datetime(self._end) - pd.to_datetime(self._start)).days
            start = pd.to_datetime(self._start, utc=True)
            end = pd.to_datetime(self._end, utc=True)
            token = self._token
            if (
                amount_days > 100
            ):  # The data provider only allows to use their screener api to get up to 100days of intraday data per api call, so a for_loop and divmod are used in order to get +100 days.
                div, remainder = divmod(amount_days, 100)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            intraday_one_ticker_100_days,
                            start=start + timedelta(days=100 * i),
                            end=start + timedelta(days=100 * (i + 1)),
                        )
                        for i in range(0, div)
                    ]
                futures = [f.result() for f in futures if not f.result().empty]
                if remainder != 0:
                    last_batch = intraday_one_ticker_100_days(
                        start + timedelta(days=(amount_days - remainder)), end
                    )
                    futures.append(last_batch)
                if len(futures) > 1:
                    df = pd.concat(futures)
                elif len(futures) == 1:
                    df = futures[0]
                else:
                    df = pd.DataFrame()
            else:
                df = intraday_one_ticker_100_days(start, end)
            if not df.empty:
                df.loc[:, "Stock"] = ticker
                df.rename(columns={"Datetime": "Date"}, inplace=True)
                df.loc[:, "Date"] = pd.to_datetime(
                    df["Date"], errors="coerce", utc=True
                )
            return df

        df = self._multithread_download_and_concat(tickers, intraday_one_ticker)
        return df.dropna(subset=["Date"])


def get_exchange_list(token):
    url = "https://eodhistoricaldata.com/api/exchanges-list/?api_token={}".format(token)
    df = pd.read_json(url)
    print(df)


def get_all_tickers_exchange(exchange, token):
    url = (
        "https://eodhistoricaldata.com/api/exchange-symbol-list/{}?api_token={}".format(
            exchange, token
        )
    )
    df = pd.read_csv(url)
    return df


def stock_screener(
    n_stocks, token, exchange, initial_offset=0, mincap=None, maxcap=None
):
    # Finds stocks by marketcap from max to min
    # initial_offset : number of stocks to skip
    # More ways to filter stocks can be found at: https://eodhistoricaldata.com/financial-apis/stock-market-screener-api/
    def one_api_call(offset, limit):
        if (mincap is None) and (maxcap is None):
            url = 'https://eodhistoricaldata.com/api/screener?api_token={}&sort=market_capitalization.desc&limit={}&offset={}&filters=[["exchange","=","{}"]]'.format(
                token, limit, offset, exchange
            )
        elif (mincap is not None) and (maxcap is None):
            url = 'https://eodhistoricaldata.com/api/screener?api_token={}&sort=market_capitalization.desc&limit={}&offset={}&filters=[["market_capitalization",">",{}],["exchange","=","{}"]]'.format(
                token, limit, offset, mincap, exchange
            )
        elif (mincap is None) and (maxcap is not None):
            url = 'https://eodhistoricaldata.com/api/screener?api_token={}&sort=market_capitalization.desc&limit={}&offset={}&filters=[["market_capitalization","<",{}],["exchange","=","{}"]]'.format(
                token, limit, offset, maxcap, exchange
            )
        else:
            url = 'https://eodhistoricaldata.com/api/screener?api_token={}&sort=market_capitalization.desc&limit={}&offset={}&filters=[["market_capitalization",">",{}],["market_capitalization","<",{}],["exchange","=","{}"]]'.format(
                token, limit, offset, mincap, maxcap, exchange
            )

        df = pd.read_json(url)
        json_struct = json.loads(df.to_json(orient="records"))
        df = pd.json_normalize(json_struct)
        if not df.empty:
            return df
        else:
            return pd.DataFrame()

    stocks = list()
    if (
        n_stocks > 100
    ):  # The data provider only allows to use their screener api to get up to a hundred stocks per api call, so a for_loop and divmod are used in order to screen +100 stocks.
        div, remainder = divmod(n_stocks, 100)
        for i in range(0, div):
            batch = one_api_call(offset=initial_offset + 100 * i, limit=100)
            stocks.append(batch)
        if remainder != 0:
            last_batch = one_api_call(
                offset=initial_offset + (n_stocks - remainder), limit=remainder
            )
            stocks.append(last_batch)
    else:
        only_batch = one_api_call(offset=initial_offset, limit=n_stocks)
        stocks.append(only_batch)
    if len(stocks) > 1:
        stocks = pd.concat(stocks).reset_index(drop=True)
    elif len(stocks) == 1:
        stocks = stocks[0]
    stocks.columns = [col.replace("data.", "") for col in stocks.columns]
    stocks.loc[:, "code"] = stocks["code"] + "." + stocks["exchange"]
    return stocks


# TODO? Add:
#   technical indicators, options, live,
#   fundamentals: index, etfs, macro indicators, bonds, goverment bonds, cds, insider trading, etc
#   upcoming earning, ipos, splits
#   bulk data full exchange 1 day
#   financial news
#   live data
#   etc
# TODO? Using requests.Session might increase performance
