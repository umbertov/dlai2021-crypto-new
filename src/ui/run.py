import hydra
import torch
from einops import rearrange
from pathlib import Path
from sys import argv

import streamlit as st

from src.common.data_utils import plot_ohlcv, plot_multi_lines
from src.common.utils import get_hydra_cfg, get_model, get_datamodule
from src.ui.ui_utils import streamlit_select_checkpoint

from copy import deepcopy

# We don't ever need gradient descent here
torch.set_grad_enabled(False)


def load_model_checkpoint(model, checkpoint_path: Path):
    return model.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


copyresult = lambda f: lambda *args, **kwargs: deepcopy(f(*args, **kwargs))

get_hydra_cfg = copyresult(st.cache(get_hydra_cfg))
get_datamodule = copyresult(st.cache(get_datamodule))

cfg = get_hydra_cfg(overrides=argv[1:])


input_columns = cfg.dataset_conf.input_columns
datamodule = get_datamodule(cfg)
train_dataloader = datamodule.train_dataloader()
# val_dataloader = datamodule.val_dataloader()[0]
train_dataset = datamodule.train_dataset
# val_dataset = datamodule.val_datasets[0]

dataframe = train_dataset.datasets[0].dataframe
input_tensors, *targets = train_dataset.datasets[0].tensors

INSTANCE_NUMBER = st.slider(
    "instance number",
    min_value=0,
    max_value=len(train_dataset.datasets[0].window_indices),
    value=0,
)
indices = train_dataset.datasets[0].window_indices[INSTANCE_NUMBER].numpy()

st.write(plot_ohlcv(dataframe.iloc[indices]))


checkpoint_path = streamlit_select_checkpoint()
model = load_model_checkpoint(get_model(cfg), checkpoint_path=checkpoint_path)
st.write(f"Created model <{cfg.model._target_}>")
st.write(f"at path {str(checkpoint_path)=} ")


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
