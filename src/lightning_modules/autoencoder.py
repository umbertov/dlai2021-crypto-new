from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
import random

import pytorch_lightning as pl

import hydra

from src.common.utils import notnone
from src.lightning_modules.base import BaseTimeSeriesModule


class TimeSeriesAutoEncoder(BaseTimeSeriesModule):
    def _reconstruction_forward(self, inputs, reconstruction=None, **_):
        if reconstruction is None:
            return None
        reconstruction_loss = self.reconstruction_loss_fn(
            reconstruction.view(-1), inputs.view(-1)
        )

        return {
            "metrics/rec_loss": reconstruction_loss,
            "reconstruction": reconstruction,
        }

    def _reconstruction_epoch_end(self, random_step, mode: str):
        if self.reconstruction_loss_fn is not None:
            reconstruction, inputs = (
                random_step["reconstruction"],
                random_step["inputs"],
            )
            if not self.hparams.model.get("channels_last", True):
                reconstruction = reconstruction.transpose(-1, -2)
                inputs = inputs.transpose(-1, -2)
            plot = regression_plot_fig(reconstruction, inputs)
            self.logger.experiment.log({f"{mode}/reconstruction_plot": plot})
