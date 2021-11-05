from typing import Callable, List
import torch
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
import pandas as pd
from glob import glob


class DataframeDataset(TensorDataset):
    dataframe: pd.DataFrame
    input_columns: List[str]
    continuous_targets: List[str]
    categorical_targets: List[str]

    def __init__(
        self, dataframe, input_columns, continuous_targets, categorical_targets
    ):
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.continuous_targets = continuous_targets
        self.categorical_targets = categorical_targets

        input_tensors = from_pandas(dataframe[input_columns])

        targets = [
            from_pandas(dataframe[columns])
            for columns in (continuous_targets, categorical_targets)
            if columns is not None
        ]

        super().__init__(input_tensors, *targets)


def from_pandas(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy())


def read_csv_dataset(
    path: str,
    reader: Callable[[str], pd.DataFrame],
    input_columns: List[str],
    continuous_targets: List[str],
    categorical_targets: List[str],
) -> Dataset:
    dataframe = reader(path)
    return DataframeDataset(
        dataframe, input_columns, continuous_targets, categorical_targets
    )


def read_csv_datasets(paths: List[str], *args, **kwargs) -> Dataset:
    """Same args as `read_csv_dataset`, except for the first one:
    - `paths`: a `List[str]` specifying a number of file paths, either absolute or relative
    """
    return ConcatDataset([read_csv_dataset(path, *args, **kwargs) for path in paths])


def read_csv_datasets_from_glob(globstr: str, *args, **kwargs) -> Dataset:
    """Same args as `read_csv_datasets`, except for taking a `globstr` argument
    instead of `paths`.
    `globstr` is a glob pattern used to match filenames.
    """
    paths = glob(globstr)
    return read_csv_datasets(paths, *args, **kwargs)


if __name__ == "__main__":
    from src.dataset_readers import example_features, read_yfinance_dataframe

    dataset = read_csv_dataset(
        "./data/yahoofinance_crypto/ADA-EUR.csv",
        example_features(read_yfinance_dataframe),
        input_columns=[
            "Log(PctChange(Close))",
            "Log(PctChange(Sma9(Close)))",
            "Log(PctChange(Sma12(Close)))",
            "Log(PctChange(Sma26(Close)))",
            "Log(Close - Sma9(Close))",
            "Log(Close - Sma12(Close))",
            "Log(Close - Sma26(Close))",
        ],
        continuous_targets=[
            "Target",
        ],
        categorical_targets=["TargetCategorical"],
    )
