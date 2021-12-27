import hydra
import torch
from einops import rearrange
import os
from pathlib import Path
import plotly.graph_objects as go
from sys import argv
from src.dataset_readers import target_categorical_adaptive_trend

import streamlit as st

from src.common.plot_utils import (
    plot_categorical_target,
    plot_categorical_tensor,
    plot_ohlcv,
    plot_multi_lines,
)
from src.common.utils import get_hydra_cfg, get_model, get_datamodule
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


checkpoint_path, run_dir, cfg, model = None, None, None, None


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_cfg_model(use_checkpoint=False, overrides=[]):
    global checkpoint_path, run_dir
    if use_checkpoint:
        checkpoint_path, run_dir = streamlit_select_checkpoint(return_run_dir=True)
        cfg = get_hydra_cfg(
            config_path=(f"{run_dir}/files"), config_name="hparams", overrides=overrides
        )
        st.write("Loaded succesfully from", run_dir)
        model = load_model_checkpoint(get_model(cfg), checkpoint_path=checkpoint_path)
        st.write(f"Created model <{cfg.model._target_}>")
    else:
        cfg = get_hydra_cfg(overrides=overrides)
        model = get_model(cfg)
    return cfg, model


use_checkpoint = sidebar.checkbox("Load Checkpoint?", value=False)

cfg, model = get_cfg_model(use_checkpoint)

input_columns = cfg.dataset_conf.input_columns
datamodule = get_datamodule(cfg)
train_dataloader = datamodule.train_dataloader()
# val_dataloader = datamodule.val_dataloader()[0]
train_dataset = datamodule.train_dataset
# val_dataset = datamodule.val_datasets[0]

DATASET_NUMBER = sidebar.slider(
    "dataset number",
    min_value=0,
    max_value=len(train_dataset.datasets),
    value=0,
)
INSTANCE_NUMBER = sidebar.slider(
    "instance number",
    min_value=0,
    max_value=len(train_dataset.datasets[DATASET_NUMBER].window_indices),
    value=0,
)
indices = train_dataset.datasets[DATASET_NUMBER].window_indices[INSTANCE_NUMBER]
st.write(indices)

full_dataframe = train_dataset.datasets[DATASET_NUMBER].dataframe
dataframe = full_dataframe.iloc[indices.numpy()]
batch = {
    k: v.unsqueeze(0)
    for k, v in train_dataset.datasets[DATASET_NUMBER][INSTANCE_NUMBER].items()
}
st.write(batch.keys())
input_tensors = batch["inputs"]
# input_tensors, *targets = [
#     t[INSTANCE_NUMBER].unsqueeze(0)
#     for t in train_dataset.datasets[DATASET_NUMBER].tensors
# ]


st.header(f"Candlestick graph for instance {INSTANCE_NUMBER}")
st.write(plot_ohlcv(dataframe))

st.plotly_chart(
    plot_multi_lines(
        **{
            colname: input_tensors[0, :, i]
            for i, colname in enumerate(cfg.dataset_conf.input_columns)
        }
    )
)

st.write("# Ground Truth plot:")
st.pyplot(
    plot_categorical_target(dataframe, target_col="TargetAdaCategorical"),
    clear_figure=True,
)

st.write(full_dataframe.TargetAdaCategorical.value_counts())

predict_out = model.predict(**batch)
st.write("# Predictions plot:")
dataframe["predictions"] = predict_out.numpy().reshape(-1)
st.pyplot(
    plot_categorical_target(dataframe, target_col="predictions"),
    clear_figure=True,
)


model_out = model(**batch)
st.write(model_out.keys())


def backtest_model_on_dataframe(cfg, dataframe, model):
    from backtesting import Backtest
    from src.evaluation.backtesting_strategies import SequenceTaggerStrategy
    from src.common.callbacks import BACKTEST_METRICS
    import pandas as pd

    bt = Backtest(
        dataframe,
        SequenceTaggerStrategy,
        cash=1000,
        commission=0.002,
        exclusive_orders=True,
    )
    stats = bt.run(
        model=model,
        cfg=cfg,
        go_short=True,
        go_long=True,
    )
    stats_df = pd.DataFrame(stats).loc[BACKTEST_METRICS]
    # stats_dict = {k: v[0] for k, v in stats_df.T.to_dict().items()}
    return stats_df


backtest_model_on_dataframe(cfg, full_dataframe.iloc[2000:5000], model)
