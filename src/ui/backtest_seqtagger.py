from backtesting.backtesting import Backtest
import hydra
import torch
from einops import rearrange
import os
from pathlib import Path
import plotly.graph_objects as go
from sys import argv

import streamlit as st

from src.common.plot_utils import plot_ohlcv, plot_multi_lines
from src.common.utils import get_hydra_cfg, get_model, get_datamodule
from src.evaluation.backtesting_strategies import (
    OptimalSequenceTaggerStrategy,
    SequenceTaggerStrategy,
)
from src.ui.ui_utils import streamlit_select_checkpoint, sidebar

from copy import deepcopy

# We don't ever need gradient descent here
torch.set_grad_enabled(False)


def load_model_checkpoint(model, checkpoint_path: Path):
    return model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


copyresult = lambda f: lambda *args, **kwargs: deepcopy(f(*args, **kwargs))

get_hydra_cfg = copyresult(get_hydra_cfg)
# get_hydra_cfg = copyresult(st.cache(get_hydra_cfg, allow_output_mutation=True))
get_datamodule = copyresult(st.cache(get_datamodule, allow_output_mutation=True))

# cfg = get_hydra_cfg(overrides=argv[1:])


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_cfg_model(use_checkpoint=False, checkpoint_path=None, run_dir=None):
    if use_checkpoint:
        assert checkpoint_path is not None
        assert run_dir is not None
        cfg = get_hydra_cfg(config_path=(f"{run_dir}/files"), config_name="hparams")
        st.write("Loaded succesfully from", run_dir)
        model = load_model_checkpoint(get_model(cfg), checkpoint_path=checkpoint_path)
        st.write(f"Created model <{cfg.model._target_}>")
    else:
        cfg = get_hydra_cfg(overrides=argv[1:])
        model = get_model(cfg)
    return cfg, model


use_checkpoint = sidebar.checkbox("Load Checkpoint?", value=False)
checkpoint_path, run_dir = streamlit_select_checkpoint(return_run_dir=True)

cfg, model = get_cfg_model(
    use_checkpoint, checkpoint_path=checkpoint_path, run_dir=run_dir
)

cfg.dataset_conf.data_path.data_path = (
    "${oc.env:PROJECT_ROOT}/data/ccxt_ohlcv/BTC-USDT*.csv"
)

input_columns = cfg.dataset_conf.input_columns
datamodule = get_datamodule(cfg)
val_dataloader = datamodule.val_dataloader()
# val_dataloader = datamodule.val_dataloader()[0]
val_dataset = datamodule.val_datasets[0]
# val_dataset = datamodule.val_datasets[0]

import site

site.main()

DATASET_NUMBER = sidebar.slider(
    "dataset number",
    min_value=0,
    max_value=len(val_dataset.datasets),
    value=0,
)
INSTANCE_NUMBER = sidebar.slider(
    "instance number",
    min_value=0,
    max_value=len(val_dataset.datasets[DATASET_NUMBER].window_indices),
    value=0,
)

indices = val_dataset.datasets[DATASET_NUMBER].window_indices[INSTANCE_NUMBER]
full_dataframe = val_dataset.datasets[DATASET_NUMBER].dataframe
dataframe = full_dataframe.iloc[indices.numpy()]
input_tensors, *targets = [
    t[INSTANCE_NUMBER].unsqueeze(0)
    for t in val_dataset.datasets[DATASET_NUMBER].tensors
]


backtest_start = sidebar.slider(
    "Backtest period start",
    min_value=0,
    max_value=len(val_dataset.datasets[DATASET_NUMBER].dataframe),
    value=2000,
    step=500,
)
backtest_length = sidebar.slider(
    "Backtest period length",
    min_value=500,
    max_value=len(val_dataset.datasets[DATASET_NUMBER].dataframe) // 10,
    value=1500,
    step=200,
)
position_size = sidebar.slider(
    "Position size %",
    min_value=0.0,
    max_value=0.99,
    value=0.99,
    step=0.1,
)
price_delta_pct = sidebar.slider(
    "Stop Loss / Take Profit % (0 = no tp/sl)",
    min_value=0.0,
    max_value=0.9,
    value=0.0,
    step=0.01,
)
if price_delta_pct == 0:
    price_delta_pct = None
backtest = Backtest(
    full_dataframe.iloc[backtest_start : backtest_start + backtest_length],
    SequenceTaggerStrategy,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True,
)
and_predictions = sidebar.slider(
    "n and predictions", min_value=0, max_value=20, value=0
)

go_short = sidebar.checkbox("Go Short?", value=False)
stats = backtest.run(
    model=model.cuda(),
    cfg=cfg,
    go_short=go_short,
    go_long=True,
    and_predictions=and_predictions,
    position_size_pct=position_size,
    price_delta_pct=price_delta_pct,
)
print(stats)
backtest.plot(results=stats, plot_return=True, plot_equity=False)

optimal_backtest = Backtest(
    full_dataframe.iloc[backtest_start : backtest_start + backtest_length],
    OptimalSequenceTaggerStrategy,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True,
)
optimal_stats = optimal_backtest.run(
    model=model.cuda(),
    cfg=cfg,
    go_short=go_short,
    go_long=True,
    and_predictions=and_predictions,
    position_size_pct=position_size,
    price_delta_pct=price_delta_pct,
)
print(optimal_stats)
optimal_backtest.plot(results=optimal_stats, plot_return=True, plot_equity=False)

launch_pdb = sidebar.checkbox("Launch ipdb?", value=False)
if launch_pdb:
    import ipdb

    ipdb.set_trace()
