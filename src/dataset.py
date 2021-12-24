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


def from_pandas(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(df.to_numpy())


class MultiTickerDataset(ConcatDataset):
    def reset(self):
        for ds in self.datasets:
            ds.reset()


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
        channels_last=True,
    ):
        assert not (minmax_scale_windows and zscore_scale_windows)
        self.dataframe = dataframe
        self.input_columns = input_columns
        self.continuous_targets = continuous_targets
        self.categorical_targets = categorical_targets
        self.name = name
        self.window_length = window_length
        self.window_skip = window_skip
        self.future_window_length = future_window_length
        self.tensor_names = ["inputs"]
        self.minmax_scale_windows = minmax_scale_windows
        self.zscore_scale_windows = zscore_scale_windows
        self.return_dicts = return_dicts
        self.channels_last = channels_last

        self._initial_window_indices = torch.tensor(
            get_window_indices(
                n_elements=len(dataframe),
                window_length=self.window_length,
                window_skip=window_skip,
                future_window_length=self.future_window_length,
            ),
            dtype=torch.long,
        )[:-1]
        self.window_indices = self._initial_window_indices.clone()

        input_tensors = self._get_input_tensors()
        targets = self._get_targets()

        super().__init__(input_tensors, *targets)

    def __getitem__(self, idx, return_df=False):
        out = super().__getitem__(idx)
        if self.return_dicts:
            out = {name: tensor for name, tensor in zip(self.tensor_names, out)}
            if return_df:
                out["dataframe"] = self.dataframe.iloc[self.window_indices[idx].numpy()]
        return out

    def reset(self):
        self.window_indices = self._initial_window_indices + torch.randint_like(
            self._initial_window_indices, low=0, high=self.window_skip - 1
        )
        input_tensors = self._get_input_tensors()
        targets = self._get_targets()
        super().__init__(input_tensors, *targets)
        pass

    def _get_input_tensors(self):
        input_tensors = from_pandas(self.dataframe[self.input_columns]).float()[
            self.window_indices
        ]
        if self.minmax_scale_windows:
            if self.minmax_scale_windows == "by_open":
                self._minmax_low = (
                    input_tensors[..., 0].min(axis=1, keepdim=True).values.unsqueeze(-1)
                )
                self._minmax_high = (
                    input_tensors[..., 0].max(axis=1, keepdim=True).values.unsqueeze(-1)
                )
            else:
                self._minmax_low = input_tensors.min(axis=1, keepdim=True).values
                self._minmax_high = input_tensors.max(axis=1, keepdim=True).values
            input_tensors = minmax_scale_tensor(
                input_tensors, self._minmax_low, self._minmax_high
            )

        if self.zscore_scale_windows:
            self._zscore_mean = input_tensors.mean(axis=1, keepdim=True)
            self._zscore_std = input_tensors.std(axis=1, keepdim=True) + 1e-20
            if self.zscore_scale_windows == "by_open":
                self._zscore_mean = self._zscore_mean[..., 0].unsqueeze(-1)
                self._zscore_std = self._zscore_std[..., 0].unsqueeze(-1)
            input_tensors = (input_tensors - self._zscore_mean) / self._zscore_std
        if not self.channels_last:
            input_tensors = input_tensors.transpose(-1, -2)
        assert input_tensors.isfinite().all()

        return input_tensors

    def _get_targets(self):
        targets = []
        if self.continuous_targets is not None:
            t = from_pandas(self.dataframe[self.continuous_targets]).float()[
                self.window_indices
            ]
            if self.minmax_scale_windows:
                t = minmax_scale_tensor(t, self._minmax_low, self._minmax_high)
            if self.zscore_scale_windows:
                t = (t - self._zscore_mean) / self._zscore_std
            targets.append(t)
            self.tensor_names.append("continuous_targets")

        if self.categorical_targets is not None:
            targets.append(
                from_pandas(
                    self.dataframe[self.categorical_targets].astype(int)
                ).long()[self.window_indices]
            )
            self.tensor_names.append("categorical_targets")

        if not self.channels_last:
            targets = [t.transpose(-1, -2) for t in targets]
        assert all(data.isfinite().all() for data in targets)
        return targets


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
    return MultiTickerDataset([i for i in datasets if i is not None])


def read_csv_datasets_from_glob(
    globstr: Union[str, List[str]], *args, **kwargs
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
