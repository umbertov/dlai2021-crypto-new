from typing import Callable, List
import torch
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
import pandas as pd
from glob import glob


class DataframeDataset(TensorDataset):
    dataframe: pd.DataFrame
    input_columns: List[str]
    target_columns: List[str]

    def __init__(self, dataframe, input_columns, target_columns):
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.target_columns = target_columns

        input_tensors = from_pandas(dataframe[input_columns])
        target_tensors = from_pandas(dataframe[target_columns])
        super().__init__(input_tensors, target_tensors)


def from_pandas(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy())


def read_csv_dataset(
    path: str,
    reader: Callable[[str], pd.DataFrame],
    input_columns: List[str],
    target_columns: List[str],
) -> Dataset:
    dataframe = reader(path)
    return DataframeDataset(dataframe, input_columns, target_columns)


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
    dataset = read_csv_dataset(
        "./data/yahoofinance_crypto/ADA-EUR.csv",
        pd.read_csv,
        ["Open", "Close"],
        ["High"],
    )
