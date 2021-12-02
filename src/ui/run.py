import hydra
import torch
from einops import rearrange
from pathlib import Path
from sys import argv

import streamlit as st

from src.common.data_utils import plot_ohlcv, plot_multi_lines
from src.common.utils import get_hydra_cfg, get_model, get_datamodule
from src.ui.ui_utils import streamlit_select_checkpoint, sidebar

from copy import deepcopy

# We don't ever need gradient descent here
torch.set_grad_enabled(False)


def load_model_checkpoint(model, checkpoint_path: Path):
    return model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


copyresult = lambda f: lambda *args, **kwargs: deepcopy(f(*args, **kwargs))

get_hydra_cfg = copyresult(st.cache(get_hydra_cfg, allow_output_mutation=True))
get_datamodule = copyresult(st.cache(get_datamodule, allow_output_mutation=True))

cfg = get_hydra_cfg(overrides=argv[1:])


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
input_tensors, *targets = [
    t[INSTANCE_NUMBER].unsqueeze(0)
    for t in train_dataset.datasets[DATASET_NUMBER].tensors
]


st.header(f"Candlestick graph for instance {INSTANCE_NUMBER}")
st.write(plot_ohlcv(dataframe))

full_dataframe_minmax_scaled = (full_dataframe - full_dataframe.Open.min()) / (
    full_dataframe.Open.max() - full_dataframe.Open.min()
)
st.header(f"Minmax Scaled version")
st.write(plot_ohlcv(full_dataframe_minmax_scaled.iloc[indices]))

ZSCORE_PERIOD = sidebar.slider("zscore period", min_value=0, max_value=500, value=20)
mean = full_dataframe.Open.rolling(ZSCORE_PERIOD).mean()
std = full_dataframe.Open.rolling(ZSCORE_PERIOD).std() * 2
full_dataframe_zscore_scaled = full_dataframe.sub(mean, axis=0).div(std, axis=0)
st.header(f"Zscore Scaled version")
st.write(plot_ohlcv(full_dataframe_zscore_scaled.iloc[indices]))


checkpoint_path = streamlit_select_checkpoint()
model = load_model_checkpoint(get_model(cfg), checkpoint_path=checkpoint_path)
st.write(f"Created model <{cfg.model._target_}>")

model_out = model(input_tensors)
st.write(model_out.keys())

st.header("Prediction vs ground truth:")
st.write(
    plot_multi_lines(
        truth=targets[0].view(-1).cpu().numpy(),
        prediction=model_out["regression_output"].view(-1).cpu().numpy(),
    )
)


# ground_truth_batch = batch.target_data.view(BATCH_SIZE, -1)
# prediction_batch = (
#     model(batch.input_data, None)["logits"].view(BATCH_SIZE, -1).cpu().numpy()
# )
#
# for y_true, y_pred in zip(ground_truth_batch[:5], prediction_batch[:5]):
#     st.write(
#         plot_multi_lines(
#             ground_truth=y_true,
#             predictions=y_pred,
#         )
#     )
