from typing import Callable, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

import src.dataset_readers as R

import hydra
from omegaconf.dictconfig import DictConfig


class DataframeDataset(TensorDataset):
    dataframe: pd.DataFrame
    input_columns: List[str]
    continuous_targets: List[str]
    categorical_targets: List[str]
    name: str

    def __init__(
        self,
        dataframe,
        input_columns,
        continuous_targets_columns,
        categorical_targets_columns,
        name,
    ):
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.continuous_targets = continuous_targets_columns
        self.categorical_targets = categorical_targets_columns
        self.name = name

        input_tensors = from_pandas(dataframe[input_columns]).float()

        targets = []
        if continuous_targets_columns is not None:
            targets.append(from_pandas(dataframe[continuous_targets_columns]).float())
        if categorical_targets_columns is not None:
            targets.append(from_pandas(dataframe[categorical_targets_columns]).long())

        super().__init__(input_tensors, *targets)


def from_pandas(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy())


def read_csv_dataset(
    path: Union[Path, str],
    reader: Callable[[str], pd.DataFrame],
    input_columns: List[str],
    continuous_targets: List[str],
    categorical_targets: List[str],
    start_date=None,
    end_date=None,
) -> Dataset:
    if not isinstance(path, Path):
        path = Path(path)
    reader_fn = (
        lambda df: (hydra.utils.instantiate(reader, df))
        if isinstance(reader, DictConfig)
        else reader
    )
    if start_date or end_date:
        reader_fn = R.Compose(
            reader_fn, R.DateRangeCut(start_date=start_date, end_date=end_date)
        )
    dataframe = reader_fn(path.absolute())
    return DataframeDataset(
        dataframe,
        input_columns,
        continuous_targets,
        categorical_targets,
        name=path.stem,
    )


def read_csv_datasets(paths: List[str], *args, **kwargs) -> Dataset:
    """Same args as `read_csv_dataset`, except for the first one:
    - `paths`: a `List[str]` specifying a number of file paths, either absolute or relative
    """
    return ConcatDataset(
        [
            read_csv_dataset(path, *args, **kwargs)
            for path in tqdm(paths, total=len(paths))
        ]
    )


def read_csv_datasets_from_glob(globstr: str, *args, **kwargs) -> Dataset:
    """Same args as `read_csv_datasets`, except for taking a `globstr` argument
    instead of `paths`.
    `globstr` is a glob pattern used to match filenames.
    """
    return read_csv_datasets(glob(globstr), *args, **kwargs)


if __name__ == "__main__":
    from src.dataset_readers import example_features

    dataset = read_csv_datasets_from_glob(
        "./data/yahoofinance_crypto/*.csv",
        example_features,
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
