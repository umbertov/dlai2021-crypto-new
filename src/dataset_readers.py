import pandas as pd
import numpy as np
from typing import Optional, Tuple


# library of functions that pre-process the raw time series and compute features.
# this would look nicer if python had (un/)currying

# it all boils down to being able to write pre-processing as a chain of basic modules


def read_yfinance_dataframe(path: str) -> pd.DataFrame:
    dataframe: pd.DataFrame = pd.read_csv(path, parse_dates=["Date"])  # type: ignore
    dataframe.index = dataframe.Date
    dataframe.drop("Date", inplace=True, axis="columns")
    return dataframe


def ColumnTrasnformer(function, name=None):
    if name is None:
        name = function.__name__

    def Transformation(input_column, target_column=None, **additional_args):
        target_column = target_column or f"{name}({input_column})"

        def apply(dataframe):
            dataframe[target_column] = function(
                dataframe[input_column], **additional_args
            )
            return dataframe

        return apply

    return Transformation


def DataframeTransformer(function, **f_kwargs):
    return lambda *args, **kwargs: lambda df: function(df, *args, **f_kwargs, **kwargs)


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
PctChange = ColumnTrasnformer(lambda col: col.pct_change() + 1, name="PctChange")
Shift = ColumnTrasnformer(lambda col: col.shift(-1), name="Shift")
LogPctChange = lambda col: Compose(PctChange(col), Log(f"PctChange({col})"))
Sma = lambda col, n: ColumnTrasnformer(lambda df: df.rolling(n).mean(), name=f"Sma{n}")(
    col
)
Bins = ColumnTrasnformer(lambda col, bins: pd.cut(col, bins), name="Bins")
BinCodes = ColumnTrasnformer(lambda col: col.values.codes, name="BinCodes")

Strip = DataframeTransformer(remove_leading_trailing_nans)


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


DateRangeCut = DataframeTransformer(
    lambda df, start_date, end_date: df.loc[start_date:end_date]
)


def get_traintest_split_readers(reader, start_date, split_date, end_date):
    train_reader = Compose(
        reader, DateRangeCut(start_date=start_date, end_date=split_date)
    )
    test_reader = Compose(
        reader, DateRangeCut(start_date=split_date, end_date=end_date)
    )
    return train_reader, test_reader


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
        Diff(base_column, f"Sma{sma_period}({base_column})", logarithmic=True),
    )


example_features = Compose(
    read_yfinance_dataframe,
    LogPctChange("Close"),
    Sma_LogPctChange_LogDiff("Close", 9),
    Sma_LogPctChange_LogDiff("Close", 12),
    Sma_LogPctChange_LogDiff("Close", 26),
    # continuous target declaration
    Shift("Log(PctChange(Close))", target_column="Target"),
    # categorical target declaration
    Shift("PctChange(Close)"),
    Strip(),
    Bins(
        "Shift(PctChange(Close))",
        bins=[0.0, 0.999, 1.001, float("inf")],
        target_column="Shift(ChangeCategorical)",
    ),
    BinCodes("Shift(ChangeCategorical)", target_column="TargetCategorical"),
)
