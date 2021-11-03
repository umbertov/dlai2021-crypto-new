import ta
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def read_yfinance_dataframe(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path, parse_dates=["Date"])
    dataframe.index = dataframe.Date
    dataframe.drop("Date", inplace=True, axis="columns")
    return dataframe


def ColumnTrasnformer(function, name=None):
    if name is None:
        name = function.__name__

    def Transformation(input_column):
        target_column = f"{name}({input_column})"

        def apply(dataframe):
            dataframe[target_column] = function(dataframe[input_column])
            return dataframe

        return apply

    return Transformation


def Compose(*transformations):
    def f(df):
        for t in transformations:
            df = t(df)
        return df

    return f


def remove_leading_trailing_nans(df: pd.DataFrame) -> pd.DataFrame:
    leading_index = pd.Series([df[col].first_valid_index() for col in df.columns]).max()
    trailing_index = pd.Series([df[col].last_valid_index() for col in df.columns]).min()
    return df.loc[leading_index:trailing_index]


Log = ColumnTrasnformer(np.log, name="Log")
PctChange = ColumnTrasnformer(lambda df: df.pct_change() + 1, name="PctChange")
LogPctChange = lambda col: Compose(PctChange(col), Log(f"PctChange({col})"))


def Diff(col1, col2, logarithmic=False):
    target_column = f"({col1} - {col2})"

    def apply(dataframe):
        dataframe[target_column] = dataframe[col1] - dataframe[col2]
        if logarithmic:
            dataframe[target_column] = np.log(
                1 + (dataframe[target_column] / dataframe[col2])
            )
        return dataframe

    return apply


def Sma(n, colname):
    return ColumnTrasnformer(lambda df: df.rolling(n).mean(), name=f"Sma{n}")(colname)


def DataframeTransformer(function, **f_kwargs):
    return lambda *args, **kwargs: lambda df: function(df, *args, **f_kwargs, **kwargs)


Strip = DataframeTransformer(remove_leading_trailing_nans)


features = lambda file_reader: Compose(
    file_reader,
    Sma(9, "Close"),
    Sma(12, "Close"),
    Sma(26, "Close"),
    LogPctChange("Close"),
    ##Log("PctChange(Close)"),
    LogPctChange("Sma9(Close)"),
    LogPctChange("Sma12(Close)"),
    LogPctChange("Sma26(Close)"),
    Diff("Close", "Sma9(Close)", logarithmic=True),
    Diff("Close", "Sma12(Close)", logarithmic=True),
    Diff("Close", "Sma26(Close)", logarithmic=True),
)


def Resample(resample_freq):
    def apply(df):
        return df.resample(resample_freq).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

    return apply
