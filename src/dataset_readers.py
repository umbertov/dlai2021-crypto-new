import pandas as pd
import numpy as np
from typing import Callable, Optional, Tuple
from ta.momentum import RSIIndicator
from typing import List, Tuple

from src.common.utils import get_hydra_cfg

DEBUG = True


# library of functions that pre-process the raw OHLCV time series and compute features.
# this would look nicer if python had syntactically good (un/)currying of functions

# it all boils down to being able to write pre-processing as a chain of basic modules


class EmptyDataFrame(Exception):
    pass


import os


def try_read_dataframe(*readers):
    def reader(path):
        dataframe = None
        for candidate_reader in readers:
            try:
                dataframe = candidate_reader(path)
            except pd.errors.EmptyDataError as e:
                with open(path, "r") as f:
                    print("path:", path)
                    print(f.read())
            except ValueError as e:
                continue
            if dataframe is not None:
                return dataframe
        raise EmptyDataFrame

    return reader


def read_binance_klines_dataframe(path: str) -> pd.DataFrame:
    dataframe: pd.DataFrame = pd.read_csv(path).iloc[:, [0, 1, 2, 3, 4, 5]]
    dataframe.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    dataframe.index = pd.to_datetime(dataframe["Date"], utc=True, unit="ms")
    dataframe.drop("Date", inplace=True, axis="columns")
    return dataframe


def read_yfinance_dataframe(path: str) -> pd.DataFrame:
    try:
        dataframe: pd.DataFrame = pd.read_csv(path, parse_dates=["Date"])  # type: ignore
    except ValueError:
        dataframe: pd.DataFrame = pd.read_csv(path, parse_dates=["Datetime"]).rename(columns={"Datetime": "Date"})  # type: ignore
    except pd.errors.EmptyDataError:
        raise EmptyDataFrame
    dataframe.index = pd.to_datetime(dataframe["Date"], utc=True)
    dataframe.drop("Date", inplace=True, axis="columns")
    return dataframe


def ColumnOp(function, name=None, **initial_args):
    if name is None:
        name = function.__name__

    def Transformation(input_column, to=None, **additional_args):
        to = to or f"{name}({input_column})"
        all_args = initial_args | additional_args

        def apply(dataframe):
            dataframe[to] = function(dataframe[input_column].copy(), **all_args)
            return dataframe

        return apply

    return Transformation


def DfOp(function, **f_kwargs):
    return lambda *args, **kwargs: lambda df: function(df, *args, **f_kwargs, **kwargs)


def Compose(*transformations):
    def f(df):
        for i, t in enumerate(transformations):
            df = t(df)
            if len(df.index) == 0:
                raise EmptyDataFrame
        return df

    return f


def remove_leading_trailing_nans(df: pd.DataFrame) -> pd.DataFrame:
    leading_index = pd.Series([df[col].first_valid_index() for col in df.columns]).max()
    trailing_index = pd.Series([df[col].last_valid_index() for col in df.columns]).min()
    if (
        len(df.index) == 0
        or leading_index > df.index[-1]
        or trailing_index < df.index[0]
    ):
        raise EmptyDataFrame
    return df.loc[leading_index:trailing_index]


Log = ColumnOp(np.log, name="Log")
PctChange = ColumnOp(lambda col: col.pct_change() + 1, name="PctChange")
Shift = ColumnOp(lambda col: col.shift(-1), name="Shift")
LogPctChange = lambda col: Compose(PctChange(col), Log(f"PctChange({col})"))
Sma = lambda col, n: ColumnOp(lambda df: df.rolling(n).mean(), name=f"Sma{n}")(col)
Std = lambda col, n: ColumnOp(lambda df: df.rolling(n).std(), name=f"Std{n}")(col)
RollingSum = ColumnOp(lambda col, window: col.rolling(window).sum())
RollingMin = ColumnOp(lambda col, window: col.rolling(window).min())
RollingMax = ColumnOp(lambda col, window: col.rolling(window).min())
Bins = ColumnOp(lambda col, bins: pd.cut(col, bins), name="Bins")
BinCodes = ColumnOp(lambda col: col.values.codes, name="BinCodes")

Strip = DfOp(remove_leading_trailing_nans)

RSI = ColumnOp(lambda df: RSIIndicator(df).rsi().dropna() / 100, name="RSI")


def zscore_norm_dataframe(df, period: int, stddev_mult: float = 2.0, by=None):
    if by is None:
        by_df = df
    else:
        by_df = df[by]
    mean = by_df.rolling(period).mean()
    std = stddev_mult * by_df.rolling(period).std()
    std[std == 0] = 1
    return df.sub(mean, axis=0).div(std, axis=0).ffill()


def ZScoreNormalize(
    colname: str, period: int, stddev_mult: float = 2.0, target_column=None, by=None
):
    def t(df):
        return zscore_norm_dataframe(df, period=period, stddev_mult=stddev_mult, by=by)

    return ColumnOp(t, name=f"Zscore{period}")(colname, to=target_column)


def Diff(col1, col2, logarithmic=False):
    target_column = f"({col1} - {col2})"
    if logarithmic:
        target_column = f"Log{target_column}"

    def apply(dataframe):
        dataframe[target_column] = dataframe[col1] - dataframe[col2]
        if logarithmic:
            dataframe[target_column] = np.log(
                1 + (dataframe[target_column] / dataframe[col2])
            )
        return dataframe

    return apply


LogDiff = lambda col1, col2: Diff(col1, col2, logarithmic=True)


def resample_ohlcv(df: pd.DataFrame, resample_freq: str) -> pd.DataFrame:
    return df.resample(resample_freq).agg(  # type: ignore
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )


Resample = lambda resample_freq: lambda df: resample_ohlcv(df, resample_freq)


DateRangeCut = DfOp(
    lambda df, start_date, end_date: remove_leading_trailing_nans(
        df.loc[start_date:end_date]
    )
)


def get_traintest_split_readers(reader, start_date, split_date, end_date):
    train_reader = Compose(
        reader, DateRangeCut(start_date=start_date, end_date=split_date)
    )
    test_reader = Compose(
        reader, DateRangeCut(start_date=split_date, end_date=end_date)
    )
    return train_reader, test_reader


def ResampleThenJoin(resample_freq, continuation, suffix=None):
    suffix = suffix or f"_{resample_freq}"

    def apply(df):
        df = df.copy()
        # first, resample ohlcv data to desired frequency
        resampled = remove_leading_trailing_nans(
            resample_ohlcv(df, resample_freq).ffill()
        )
        # add suffix to columns to avoid naming conflicts when re-joining
        resampled = resampled.rename(columns=lambda colname: f"{colname}{suffix}")
        # apply some processing to resampled dataframe
        resampled = continuation(resampled)
        # we obtain the originary timeframe by looking at the TimeDelta between rows
        original_tf = df.index[1] - df.index[0]
        # resample dataframe back to the originary timeframe, forward-filling information
        # from the higher timeframe
        resampled = resampled.resample(original_tf).ffill()
        # join the old columns with the newly created ones
        joined = df.join(resampled)
        return remove_leading_trailing_nans(joined)

    return apply


def DebugIfNan():
    if not DEBUG:
        return lambda x: x

    def f(df):
        if df.isna().any().any():

            import ipdb

            ipdb.set_trace()
        return df

    return f


def AddShiftedColumns(shift_amt: int, columns: List[str]):
    def f(df):
        for col in columns:
            df[f"Shift{shift_amt}({col})"] = df[col].shift(shift_amt)
        return df

    return f


def Sma_LogPctChange_LogDiff(base_column, sma_period):
    """
    Adds columns:
        - SmaN(base_column)
        - PctChange(base_column)
        - Log(PctChange(base_column))
        - Log(Close - SmaN(base_column))
    """
    return Compose(
        Sma(base_column, sma_period),
        LogPctChange(f"Sma{sma_period}({base_column})"),
        LogDiff(base_column, f"Sma{sma_period}({base_column})"),
    )


def Multi_Sma_LogPctChange_LogDiff(colname, sma_lengths: List[int]):
    return Compose(
        *[Sma_LogPctChange_LogDiff(colname, sma_len) for sma_len in sma_lengths]
    )


def rolling_sums(windows):
    return Compose(
        *[
            RollingSum(
                f"Log(PctChange(Open))",
                window=window,
                to=f"LogCumReturn{window}(Open)",
            )
            for window in windows
        ]
    )


def features_from_1h(df):
    return Compose(
        ### Features from another timeframe
        ### (1h timeframe)
        ResampleThenJoin(
            "1h",
            Compose(
                LogPctChange(f"Open_1h"),
                AddShiftedColumns(1, [f"Open_1h"]),
                Multi_Sma_LogPctChange_LogDiff(f"Open_1h", [9, 12, 26]),
                Strip(),
            ),
        ),
        LogDiff("Open", f"Sma9(Open_1h)"),
        LogDiff("Open", f"Sma12(Open_1h)"),
        LogDiff("Open", f"Sma26(Open_1h)"),
        Strip(),
        lambda df: df.dropna(),
        DebugIfNan(),
    )(df)


def logreturns_target():
    return Shift(f"Log(PctChange(Open))", to="Target")


def returns_target():
    return Shift(f"PctChange(Open)", to="Target")


def cumlogreturn_target(periods=5):
    def f(df):
        logreturns = df["Log(PctChange(Open))"]
        rollingsum = logreturns.rolling(periods).sum().shift(-periods)
        target = rollingsum.shift(-1)
        df["Target"] = target
        return df

    return f


def bin_column(column, bins):
    return Compose(
        Bins(column, bins=bins),
        BinCodes(f"Bins({column})", to=f"{column}Categorical"),
    )


def qbin_column(column, q):
    return Compose(
        QBins(column, q=q),
        BinCodes(f"QBins({column})", to=f"{column}Categorical"),
    )


def bin_target(bins):
    return bin_column("Target", bins)


def qbin_target(q):
    return qbin_column("Target", q)


def qbins(df, q):
    _, bins = pd.qcut(df, q, retbins=True)
    bins[0] = float("-inf")
    bins[-1] = float("inf")
    return pd.cut(df, bins=bins)


QBins = ColumnOp(qbins, name="QBins")


def zscore_normalize_target(period=200):
    return ZScoreNormalize("Target", period=period, target_column="TargetNormed")


feature_set_1 = Compose(
    ### features from default timeframe (5min)
    # logarithmic pct change from last row
    LogPctChange(f"Open"),
    #
    rolling_sums(windows=[12, 48, 488]),
    RSI(f"Open"),
    # simple moving averages + logdiff betweek column and sma + log pct change in sma
    Multi_Sma_LogPctChange_LogDiff("Open", [9, 12, 26]),
    features_from_1h,
)

ohlc4_mean = lambda df: (df.Open + df.Close + df.High + df.Low) / 4


def LogOHLC():
    return Compose(*[Log(col) for col in ["Open", "High", "Low", "Close"]])


def red_or_green_candle(df):
    """Adds a column where we have True for green, and False for red candles"""
    df["RedOrGreen"] = (df.Close - df.Open) > 0
    return df


from dataclasses import dataclass


@dataclass
class CandleCenter:
    OHLC: Tuple[str, str, str, str] = ("Open", "High", "Low", "Close")
    close: str = "Close"

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        OHLC = list(self.OHLC)
        normed = np.log(df[OHLC]).sub(np.log(df[OHLC]), axis=0)
        return normed

    def decode(self, normed: pd.DataFrame) -> pd.DataFrame:
        OHLC = list(self.OHLC)
        recons_open = normed[self.close].cumsum()
        reconstructed = np.exp(normed[OHLC].add(recons_open.shift(1), axis=0))
        return reconstructed


def center_candles(df, scaler=CandleCenter()):
    target_cols = [f"Centered{col}" for col in scaler.OHLC]
    centered = scaler.encode(df)
    # perform z score scaling (scale to be 0 mean and 1 variance)
    # mean is already 0
    std = centered.rolling(200).std()
    std[std == 0] = 1
    centered = (centered) / std
    # (Bullshit:) # scale candles to have range in [0,1], mean 0.5 and variance 0.25
    # centered = 0.5 + centered * 0.125
    df[target_cols] = centered
    return df


def NormOHLC4(cols):
    def f(df):
        ohlc4 = ohlc4_mean(df)
        ohlc4[ohlc4 == 0] = 1
        for col in cols:
            df[f"NormOHLC4({col})"] = df[col] / ohlc4
        return df

    return f


def FutureMeanStdTarget(period=10):
    def future_mean_logvar_target(df):
        ohlc4 = ohlc4_mean(df)
        mean, std = ohlc4.rolling(period).mean(), ohlc4.rolling(period).std()
        df["FutureMean"] = mean.shift(-period)
        df["FutureStd"] = std.shift(-period)
        return df

    return future_mean_logvar_target


feature_set_2 = Compose(
    center_candles,
    Strip(),
    LogOHLC(),
    RSI(f"Close"),
    Sma("Open", 9),
    Std("Open", 9),
    Sma("Open", 26),
    Std("Open", 26),
    Strip(),
    NormOHLC4(
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Sma9(Open)",
            "Std9(Open)",
            "Sma26(Open)",
            "Std26(Open)",
        ]
    ),
    Strip(),
    red_or_green_candle,
    FutureMeanStdTarget(period=10),
    Shift("RedOrGreen", to="FutureRedOrGreen"),
    Shift("Open", to="FutureOpen"),
    Shift("High", to="FutureHigh"),
    Shift("Low", to="FutureLow"),
    Shift("Close", to="FutureClose"),
    Shift("Log(Open)", to="Log(FutureOpen)"),
    Shift("Log(High)", to="Log(FutureHigh)"),
    Shift("Log(Low)", to="Log(FutureLow)"),
    Shift("Log(Close)", to="Log(FutureClose)"),
    Shift("CenteredOpen", to="CenteredFutureOpen"),
    Shift("CenteredHigh", to="CenteredFutureHigh"),
    Shift("CenteredLow", to="CenteredFutureLow"),
    Shift("CenteredClose", to="CenteredFutureClose"),
    Strip(),
)

OHLC = ["Open", "High", "Low", "Close"]


def target_volatility_classification(
    df, trend_period=20, target_col="TargetCategoricalVolatility", std_mult=3.0
):
    returns = df.Close.pct_change()
    std_returns = returns.rolling(trend_period).std()
    df["RollingReturnVolatility"] = std_returns

    upper_barrier = df.Close * (1 + std_returns * std_mult)
    lower_barrier = df.Close * (1 - std_returns * std_mult)

    future_prices = df.Close.shift(-trend_period)
    future_maxes = future_prices.rolling(trend_period).max()
    future_mins = future_prices.rolling(trend_period).min()
    buy_signals = (future_maxes > upper_barrier) & ~(future_mins < lower_barrier)
    sell_signals = (future_mins < lower_barrier) & ~(future_maxes < upper_barrier)

    buy_signals, sell_signals = buy_signals.astype(int), sell_signals.astype(int)

    assert not (buy_signals & sell_signals).any()

    # combine signals in the [0,1,2] range
    signals = buy_signals - sell_signals + 1

    if target_col is None:
        return signals
    df[target_col] = signals
    return df


def target_categorical_trend(
    df, trend_period=10, target_col="TargetCategorical", alpha=0.01
):
    ohlc4 = ohlc4_mean(df)
    mean_before = ohlc4.rolling(trend_period).mean()
    mean_after = ohlc4.shift(-trend_period)

    ratio = mean_after / mean_before - 1

    # greater = (mean_after > mean_before * (1 + alpha)).astype(int)
    # smaller = (mean_after < mean_before * (1 - alpha)).astype(int)
    greater = (ratio > alpha).astype(int)
    smaller = (ratio < -alpha).astype(int)

    assert not (greater & smaller).any()

    # combine signals in the [0,1,2] range
    signals = greater - smaller + 1

    if target_col is None:
        return signals
    df[target_col] = signals
    return df


def target_categorical_adaptive_trend(
    df, trend_period=10, target_col="TargetAdaCategorical", std_mult=1.0
):
    ohlc4 = ohlc4_mean(df)
    df["OHLC4"] = ohlc4
    mean_before = ohlc4.rolling(trend_period).mean()
    mean_after = ohlc4.shift(-trend_period)

    ratio = mean_after / mean_before - 1
    alpha = std_mult * ratio.rolling(2 * trend_period).std()

    greater = (ratio > alpha).astype(int)
    smaller = (ratio < -alpha).astype(int)

    assert not (greater & smaller).any()

    # combine signals in the [0,1,2] range
    signals = greater - smaller + 1

    if target_col is None:
        return signals
    df[target_col] = signals
    return df


def feature_set_2_trendprediction_reader(
    resample="5min",
    trend_period=10,
    target_col="TargetCategorical",
    alpha=0.01,
    std_mult=1.0,
):
    return Compose(
        try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
        Resample(resample),
        feature_set_2,
        lambda df: target_categorical_trend(
            df, trend_period=trend_period, target_col=target_col, alpha=alpha
        ),
        lambda df: target_categorical_adaptive_trend(
            df,
            trend_period=trend_period,
            target_col="TargetAdaCategorical",
            std_mult=std_mult,
        ),
        Strip(),
        lambda df: df.ffill(),
    )


def feature_set_2_reader(resample="5min"):
    return Compose(
        try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
        Resample(resample),
        feature_set_2,
        Strip(),
        lambda df: df.ffill(),
    )


example_reader = lambda: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    feature_set_1,
    returns_target(),
    bin_target(bins=[0.0, 0.999, 1.001, float("inf")]),
    Log("Target", to="Target"),
    zscore_normalize_target(period=200),
    Strip(),
    DebugIfNan(),
)


qbin_reader = lambda q: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    feature_set_1,
    logreturns_target(),
    qbin_column("Target", q=q),
    zscore_normalize_target(period=200),
    Strip(),
    DebugIfNan(),
)


qbin_cumreturn_reader = lambda q: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    feature_set_1,
    cumlogreturn_target(5),
    qbin_column("Target", q=q),
    zscore_normalize_target(period=50),
    Strip(),
    DebugIfNan(),
)


bin_cumreturn_reader = lambda bins: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    feature_set_1,
    cumlogreturn_target(5),
    bin_column("Target", bins=bins),
    zscore_normalize_target(period=10),
    Strip(),
    DebugIfNan(),
)


def minmax_scale_df(df):
    low, high = df.min(), df.max()
    return (df - low) / (high - low)


def zscore_normalize_columns(
    columns: List[str], period: int, stddev_mult: float = 2, by=None
):
    normalizers = [
        ZScoreNormalize(colname, period, stddev_mult, by=by) for colname in columns
    ]
    return Compose(*normalizers)


MinMaxScaler = ColumnOp(minmax_scale_df, name="MinMax")

minmax_ohlcv = Compose(
    MinMaxScaler("Open"),
    MinMaxScaler("High"),
    MinMaxScaler("Low"),
    MinMaxScaler("Close"),
    Shift("MinMax(Close)", to="FutureClose"),
)

minmax_reader = lambda: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    minmax_ohlcv,
)


def Debug():
    def f(df):
        import ipdb

        ipdb.set_trace()
        return df

    return f


# zscore_by_open = lambda period: ZscoreBy(
#    columns=["Open", "High", "Low", "Close"], period=period, by="Open"
# )


from src.common.data_utils import ohlc_cols


def zscore_by_open(period, std_mult=1.0):
    def f(df):
        mean = df.Open.rolling(period).mean()
        std = df.Open.rolling(period).std()
        for col in ohlc_cols:
            df[f"Zscore{period}({col})"] = (
                df[col].sub(mean, axis=0).div(std * std_mult + 1e-10, axis=0)
            )
        return df

    return f


def zscore_by_ohlc4(period, std_mult=1.0):
    def f(df):
        ohlc4 = sum((df.Open, df.Close, df.High, df.Low)) / 4
        mean = ohlc4.rolling(period).mean()
        std = ohlc4.rolling(period).std()
        for col in ohlc_cols:
            df[f"Zscore{period}({col})"] = (
                df[col].sub(mean, axis=0).div(std * std_mult + 1e-10, axis=0)
            )
        return df

    return f


def zscore_by_open_reader(resample, zscore_period=100, std_mult=1.0):
    return Compose(
        try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
        Resample(resample),
        LogPctChange(col="Open"),
        zscore_by_open(period=zscore_period, std_mult=std_mult),
        Shift(f"Zscore{zscore_period}(Open)", to="FutureOpen"),
        Shift(f"Zscore{zscore_period}(High)", to="FutureHigh"),
        Shift(f"Zscore{zscore_period}(Low)", to="FutureLow"),
        Shift(f"Zscore{zscore_period}(Close)", to="FutureClose"),
        Shift("Log(PctChange(Open))", to="FutureLogReturn"),
        Strip(),
        lambda df: df.dropna(),
        DebugIfNan(),
    )


def zscore_by_ohlc4_reader(resample, zscore_period=100, std_mult=1.0):
    return Compose(
        try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
        Resample(resample),
        LogPctChange(col="Open"),
        zscore_by_ohlc4(period=zscore_period, std_mult=std_mult),
        Shift(f"Zscore{zscore_period}(Open)", to="FutureOpen"),
        Shift(f"Zscore{zscore_period}(High)", to="FutureHigh"),
        Shift(f"Zscore{zscore_period}(Low)", to="FutureLow"),
        Shift(f"Zscore{zscore_period}(Close)", to="FutureClose"),
        Shift("Log(PctChange(Open))", to="FutureLogReturn"),
        Strip(),
        lambda df: df.dropna(),
        DebugIfNan(),
    )


zscore_reader = lambda normalize_colnames: Compose(
    example_reader,
    zscore_normalize_columns(normalize_colnames, period=20),
    zscore_normalize_columns(normalize_colnames, period=50),
    zscore_normalize_columns(normalize_colnames, period=100),
    zscore_normalize_columns(normalize_colnames, period=200),
    Strip(),
    DebugIfEmpty,
)

zscore_reader_qbins = lambda normalize_colnames, q: Compose(
    qbin_reader(q=q),
    zscore_normalize_columns(normalize_colnames, period=20),
    zscore_normalize_columns(normalize_colnames, period=50),
    zscore_normalize_columns(normalize_colnames, period=100),
    zscore_normalize_columns(normalize_colnames, period=200),
    Strip(),
    DebugIfEmpty,
)

zscore_cumreturn_qbin_reader = lambda normalize_colnames, q: Compose(
    qbin_cumreturn_reader(q=q),
    zscore_normalize_columns(normalize_colnames, period=20),
    zscore_normalize_columns(normalize_colnames, period=50),
    zscore_normalize_columns(normalize_colnames, period=100),
    zscore_normalize_columns(normalize_colnames, period=200),
    Strip(),
    # lambda df: df.ffill(),
    DebugIfEmpty,
)

zscore_cumreturn_gtzero_reader = lambda normalize_colnames: Compose(
    bin_cumreturn_reader(bins=GTZERO_BINS),
    minmax_ohlcv,
    zscore_normalize_columns(normalize_colnames, period=20),
    zscore_normalize_columns(normalize_colnames, period=50),
    zscore_normalize_columns(normalize_colnames, period=100),
    zscore_normalize_columns(normalize_colnames, period=200),
    Strip(),
    DebugIfNan(),
    # lambda df: df.ffill(),
    DebugIfEmpty,
)


def goodtargets(
    trend_period=10,
    target_col="TargetCategorical",
    alpha=0.01,
    std_mult=1.0,
):
    return Compose(
        lambda df: target_categorical_trend(
            df, trend_period=trend_period, target_col=target_col, alpha=alpha
        ),
        lambda df: target_categorical_adaptive_trend(
            df,
            trend_period=trend_period,
            target_col="TargetAdaCategorical",
            std_mult=std_mult,
        ),
        lambda df: target_volatility_classification(
            df,
            trend_period=trend_period,
            target_col="TargetCategoricalVolatility",
            std_mult=std_mult,
        ),
        Strip(),
        lambda df: df.ffill(),
    )


def goodfeatures(
    zscore_periods=[10, 20, 30, 50, 100, 200, 2000],
    scaled_columns=["Open", "High", "Low", "Close", "Volume", "Log(PctChange(Close))"],
):
    return Compose(
        LogPctChange("Close"),
        *[
            zscore_normalize_columns(scaled_columns, period=period)
            for period in zscore_periods
        ],
        RSI("Close"),
    )


def sma(period, column):
    def f(df):
        df[f"Sma({column})"] = df[column].rolling(period).mean()
        return df

    return f


def smas(periods: list[int], columns=list[str]):
    return Compose(*[sma(period, column) for period in periods for column in columns])


def goodfeatures_reader(
    resample="5min",
    trend_period=10,
    target_col="TargetCategorical",
    alpha=0.01,
    std_mult=1.0,
    zscore_periods=[10, 20, 30, 50, 100, 200, 2000],
    scaled_columns=["Open", "High", "Low", "Close", "Volume", "Log(PctChange(Close))"],
):
    return Compose(
        try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
        Resample(resample),
        goodtargets(
            trend_period=trend_period,
            target_col=target_col,
            alpha=alpha,
            std_mult=std_mult,
        ),
        goodfeatures(zscore_periods=zscore_periods, scaled_columns=scaled_columns),
        Strip(),
        lambda df: df.ffill(),
    )


GTZERO_BINS = [float("-inf"), 0, float("inf")]


def DebugIfEmpty(df):
    if len(df.index) == 0:
        import ipdb

        ipdb.set_trace()
    return df


if __name__ == "__main__":
    cfg = get_hydra_cfg("../../conf")

    input_columns = cfg.dataset_conf.input_columns
    reader = Compose(
        example_reader,
        zscore_normalize_columns(input_columns, period=20),
        zscore_normalize_columns(input_columns, period=50),
        zscore_normalize_columns(input_columns, period=100),
        Strip(),
    )
    df = reader("./data/yahoofinance_crypto/BTC-USD.2021-08-30.2021-10-28.csv")
