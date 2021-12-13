import pandas as pd
import numpy as np
from typing import Callable, Optional, Tuple
from ta.momentum import RSIIndicator

from src.common.utils import get_hydra_cfg

DEBUG = True


# library of functions that pre-process the raw time series and compute features.
# this would look nicer if python had (un/)currying

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
    except EmptyDataError:
        raise EmptyDataFrame
    dataframe.index = pd.to_datetime(dataframe["Date"], utc=True)
    dataframe.drop("Date", inplace=True, axis="columns")
    return dataframe


def ColumnTransformer(function, name=None):
    if name is None:
        name = function.__name__

    def Transformation(
        input_column, target_column=None, is_input_feature=False, **additional_args
    ):
        target_column = target_column or f"{name}({input_column})"
        if is_input_feature:
            target_column = f"Feature{target_column}"

        def apply(dataframe):
            dataframe[target_column] = function(
                dataframe[input_column].copy(), **additional_args
            )
            return dataframe

        return apply

    return Transformation


def DataframeTransformer(function, **f_kwargs):
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


Log = ColumnTransformer(np.log, name="Log")
PctChange = ColumnTransformer(lambda col: col.pct_change() + 1, name="PctChange")
Shift = ColumnTransformer(lambda col: col.shift(-1), name="Shift")
LogPctChange = lambda col: Compose(PctChange(col), Log(f"PctChange({col})"))
Sma = lambda col, n: ColumnTransformer(lambda df: df.rolling(n).mean(), name=f"Sma{n}")(
    col
)
Std = lambda col, n: ColumnTransformer(lambda df: df.rolling(n).std(), name=f"Std{n}")(
    col
)
RollingSum = ColumnTransformer(lambda col, window: col.rolling(window).sum())
RollingMin = ColumnTransformer(lambda col, window: col.rolling(window).min())
RollingMax = ColumnTransformer(lambda col, window: col.rolling(window).min())
Bins = ColumnTransformer(lambda col, bins: pd.cut(col, bins), name="Bins")
BinCodes = ColumnTransformer(lambda col: col.values.codes, name="BinCodes")

Strip = DataframeTransformer(remove_leading_trailing_nans)

RSI = ColumnTransformer(lambda df: RSIIndicator(df).rsi().dropna() / 100, name="RSI")


def zscore_norm_dataframe(df, period: int, stddev_mult: float = 2.0, by=None):
    if by is None:
        by_df = df
    else:
        by_df = df[by]
    mean = by_df.rolling(period).mean()
    std = 1e-30 + stddev_mult * by_df.rolling(period).std()
    return df.sub(mean, axis=0).div(std, axis=0).ffill()


def ZScoreNormalize(
    colname: str, period: int, stddev_mult: float = 2.0, target_column=None, by=None
):
    def transformer(df):
        return zscore_norm_dataframe(df, period=period, stddev_mult=stddev_mult, by=by)

    return ColumnTransformer(transformer, name=f"Zscore{period}")(
        colname, target_column=target_column
    )


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


DateRangeCut = DataframeTransformer(
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


def AddShiftedColumns(shift_amt: int, columns: list[str]):
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


def Multi_Sma_LogPctChange_LogDiff(colname, sma_lengths: list[int]):
    return Compose(
        *[Sma_LogPctChange_LogDiff(colname, sma_len) for sma_len in sma_lengths]
    )


def rolling_sums(windows):
    return Compose(
        *[
            RollingSum(
                f"Log(PctChange(Open))",
                window=window,
                target_column=f"LogCumReturn{window}(Open)",
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
    return Shift(f"Log(PctChange(Open))", target_column="Target")


def returns_target():
    return Shift(f"PctChange(Open)", target_column="Target")


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
        BinCodes(f"Bins({column})", target_column=f"{column}Categorical"),
    )


def qbin_column(column, q):
    return Compose(
        QBins(column, q=q),
        BinCodes(f"QBins({column})", target_column=f"{column}Categorical"),
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


QBins = ColumnTransformer(qbins, name="QBins")


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
    OHLC: tuple[str, str, str, str] = ("Open", "High", "Low", "Close")
    close: str = "Close"

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        normed = np.log(df[list(self.OHLC)]).sub(np.log(df[self.OHLC[0]]), axis=0)
        return normed

    def decode(self, normed: pd.DataFrame) -> pd.DataFrame:
        recons_open = normed[self.close].cumsum()
        reconstructed = np.exp(
            normed[list(self.OHLC)].add(recons_open.shift(1), axis=0)
        )
        return reconstructed


def center_candles(df, scaler=CandleCenter()):
    target_cols = [f"Centered{col}" for col in scaler.OHLC]
    centered = scaler.encode(df)
    # perform z score scaling (scale to be 0 mean and 1 variance)
    # mean is already 0
    std = centered.rolling(200).std()
    std[std == 0] = 1
    centered = (centered) / std
    # scale candles to have range in [0,1], mean 0.5 and variance 0.25
    centered = 0.5 + centered * 0.125
    df[target_cols] = centered
    return df


def NormOHLC4(cols):
    def f(df):
        ohlc4 = ohlc4_mean(df)
        for col in cols:
            df[f"NormOHLC4({col})"] = df[col] / (ohlc4 + 1e-12)
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
    Shift("RedOrGreen", target_column="FutureRedOrGreen"),
    Shift("Open", target_column="FutureOpen"),
    Shift("High", target_column="FutureHigh"),
    Shift("Low", target_column="FutureLow"),
    Shift("Close", target_column="FutureClose"),
    Shift("Log(Open)", target_column="Log(FutureOpen)"),
    Shift("Log(High)", target_column="Log(FutureHigh)"),
    Shift("Log(Low)", target_column="Log(FutureLow)"),
    Shift("Log(Close)", target_column="Log(FutureClose)"),
    Shift("CenteredOpen", target_column="CenteredFutureOpen"),
    Shift("CenteredHigh", target_column="CenteredFutureHigh"),
    Shift("CenteredLow", target_column="CenteredFutureLow"),
    Shift("CenteredClose", target_column="CenteredFutureClose"),
    Strip(),
)


def target_categorical_trend(
    df, trend_period=10, target_col="TargetCategorical", alpha=0.01
):
    ohlc4 = ohlc4_mean(df)
    mean = ohlc4.rolling(trend_period).mean()
    mean_next = ohlc4.shift(-trend_period)

    greater = (mean_next > mean * (1 + alpha)).astype(int)
    smaller = (mean_next < mean * (1 - alpha)).astype(int)

    assert not (greater & smaller).any()

    # combine signals in the [0,1,2] range
    signals = greater - smaller + 1

    if target_col is None:
        return signals
    df[target_col] = signals
    return df


def feature_set_2_trendprediction_reader(
    resample="5min", trend_period=10, target_col="TargetCategorical", alpha=0.01
):
    return Compose(
        try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
        Resample(resample),
        feature_set_2,
        lambda df: target_categorical_trend(
            df, trend_period=trend_period, target_col=target_col, alpha=alpha
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


resampling_feature_set_2_reader = lambda resample: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    Resample(resample),
    lambda df: df.ffill(),
    DebugIfNan(),
    feature_set_2,
    Strip(),
)

example_reader = lambda: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    feature_set_1,
    returns_target(),
    bin_target(bins=[0.0, 0.999, 1.001, float("inf")]),
    Log("Target", target_column="Target"),
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
    columns: list[str], period: int, stddev_mult: float = 2, by=None
):
    normalizers = [
        ZScoreNormalize(colname, period, stddev_mult, by=by) for colname in columns
    ]
    return Compose(*normalizers)


MinMaxScaler = ColumnTransformer(minmax_scale_df, name="MinMax")

minmax_ohlcv = Compose(
    MinMaxScaler("Open"),
    MinMaxScaler("High"),
    MinMaxScaler("Low"),
    MinMaxScaler("Close"),
)

minmax_reader = lambda: Compose(
    try_read_dataframe(read_yfinance_dataframe, read_binance_klines_dataframe),
    minmax_ohlcv,
    Shift("MinMax(Close)", target_column="FutureClose"),
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
        Shift(f"Zscore{zscore_period}(Open)", target_column="FutureOpen"),
        Shift(f"Zscore{zscore_period}(High)", target_column="FutureHigh"),
        Shift(f"Zscore{zscore_period}(Low)", target_column="FutureLow"),
        Shift(f"Zscore{zscore_period}(Close)", target_column="FutureClose"),
        Shift("Log(PctChange(Open))", target_column="FutureLogReturn"),
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
        Shift(f"Zscore{zscore_period}(Open)", target_column="FutureOpen"),
        Shift(f"Zscore{zscore_period}(High)", target_column="FutureHigh"),
        Shift(f"Zscore{zscore_period}(Low)", target_column="FutureLow"),
        Shift(f"Zscore{zscore_period}(Close)", target_column="FutureClose"),
        Shift("Log(PctChange(Open))", target_column="FutureLogReturn"),
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
