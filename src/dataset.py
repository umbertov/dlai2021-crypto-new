from typing import Callable, List, Optional, Union
from pathlib import Path
import torch
from torch import tensor
from torch.utils.data import TensorDataset, ConcatDataset, Dataset
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

import src.dataset_readers as R

import hydra
from omegaconf.dictconfig import DictConfig


def minmax_scale_tensor(tensor, low, high):
    assert ((high - low) > 0).all()
    return (tensor - low) / (high - low)


def get_window_indices(n_elements, window_length, window_skip, future_window_length=0):
    return [
        range(i, i + window_length)
        for i in range(n_elements - window_length - future_window_length)
    ][::window_skip]


class DataframeDataset(TensorDataset):
    dataframe: pd.DataFrame
    input_columns: List[str]
    continuous_targets: List[str]
    categorical_targets: List[str]
    name: str
    window_length: int

    def __init__(
        self,
        dataframe,
        input_columns,
        continuous_targets,
        categorical_targets,
        name,
        window_length=1,
        window_skip=1,
        future_window_length=0,
        return_dicts=False,
        minmax_scale_windows=False,
        zscore_scale_windows=False,
        clamp_values=None,
    ):
        assert not (minmax_scale_windows and zscore_scale_windows)
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.continuous_targets = continuous_targets
        self.categorical_targets = categorical_targets
        self.name = name
        self.window_length = window_length
        self.future_window_length = future_window_length
        self.tensor_names = ["inputs"]
        self.minmax_scale_windows = minmax_scale_windows
        self.zscore_scale_windows = zscore_scale_windows
        self.return_dicts = return_dicts

        window_indices = torch.tensor(
            get_window_indices(
                n_elements=len(dataframe),
                window_length=self.window_length,
                window_skip=window_skip,
                future_window_length=self.future_window_length,
            ),
            dtype=torch.long,
        )
        self.window_indices = window_indices
        input_tensors = from_pandas(dataframe[input_columns]).float()[window_indices]
        if minmax_scale_windows:
            if minmax_scale_windows == "by_open":
                low = (
                    input_tensors[..., 0].min(axis=1, keepdim=True).values.unsqueeze(-1)
                )
                high = (
                    input_tensors[..., 0].max(axis=1, keepdim=True).values.unsqueeze(-1)
                )
            else:
                low = input_tensors.min(axis=1, keepdim=True).values
                high = input_tensors.max(axis=1, keepdim=True).values
            input_tensors = minmax_scale_tensor(input_tensors, low, high)

        if zscore_scale_windows:
            mean = input_tensors.mean(axis=1, keepdim=True)
            std = input_tensors.std(axis=1, keepdim=True)
            if zscore_scale_windows == "by_open":
                mean = mean[..., 0].unsqueeze(-1)
                std = std[..., 0].unsqueeze(-1)
            input_tensors = (input_tensors - mean) / std

        targets = []
        if continuous_targets is not None:
            t = from_pandas(dataframe[continuous_targets]).float()[window_indices]
            if minmax_scale_windows:
                t = minmax_scale_tensor(t, low, high)
            if zscore_scale_windows:
                t = (t - mean) / std
            targets.append(t)
            self.tensor_names.append("continuous_targets")

        if categorical_targets is not None:
            targets.append(
                from_pandas(dataframe[categorical_targets].astype(int)).long()[
                    window_indices
                ]
            )
            self.tensor_names.append("categorical_targets")
        if future_window_length > 0:
            future_window_indices = torch.tensor(
                [
                    range(i + window_length, i + window_length + future_window_length)
                    for i in range(
                        len(dataframe) - window_length - future_window_length
                    )
                ][::window_skip],
                dtype=torch.long,
            )
            self.future_window_indices = future_window_indices
            t = from_pandas(dataframe[continuous_targets]).float()[
                future_window_indices
            ]
            if minmax_scale_windows:
                t = minmax_scale_tensor(t, low, high)
            if zscore_scale_windows:
                t = (t - mean) / std
            targets.append(t)
            self.tensor_names.append("future_continuous_targets")

        assert input_tensors.isfinite().all()
        assert all(data.isfinite().all() for data in targets)

        if clamp_values is not None:
            for t in [input_tensors] + targets:
                torch.clamp(t, clamp_values.min, clamp_values.max, out=t)

        super().__init__(input_tensors, *targets)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if self.return_dicts:
            return {name: tensor for name, tensor in zip(self.tensor_names, out)}
        return out


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
    **kwargs,
) -> Optional[Dataset]:
    if not isinstance(path, Path):
        path = Path(path)
    reader_fn = (
        lambda datapath: hydra.utils.instantiate(reader)(datapath)
        if isinstance(reader, DictConfig)
        else reader
    )
    if start_date or end_date:
        reader_fn = R.Compose(
            reader_fn, R.DateRangeCut(start_date=start_date, end_date=end_date)
        )
    try:
        dataframe = reader_fn(path.absolute())
    except R.EmptyDataFrame as e:
        return None
    return DataframeDataset(
        dataframe,
        input_columns,
        continuous_targets,
        categorical_targets,
        name=path.stem,
        **kwargs,
    )


def read_csv_datasets(paths: List[str], *args, **kwargs) -> Dataset:
    """Same args as `read_csv_dataset`, except for the first one:
    - `paths`: a `List[str]` specifying a number of file paths, either absolute or relative
    """
    datasets = [
        read_csv_dataset(path, *args, **kwargs) or None
        for path in tqdm(paths, total=len(paths))
    ]
    return ConcatDataset([i for i in datasets if i is not None])


def read_csv_datasets_from_glob(
    globstr: Union[str, list[str]], *args, **kwargs
) -> Dataset:
    """Same args as `read_csv_datasets`, except for taking a `globstr` argument
    instead of `paths`.
    `globstr` is a glob pattern used to match filenames.
    """
    if isinstance(globstr, str):
        paths = list(glob(globstr))
    else:
        paths = [path for pattern in globstr for path in glob(pattern)]
    return read_csv_datasets(paths, *args, **kwargs)


if __name__ == "__main__":
    from src.dataset_readers import example_features

    dataset = read_csv_datasets_from_glob(
        "./data/yahoofinance_crypto/*.csv",
        reader=example_features,
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
