from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
import random

import pytorch_lightning as pl

import hydra

from src.common.utils import notnone
from .autoencoder import TimeSeriesAutoEncoder


class TimeSeriesVAE(TimeSeriesAutoEncoder):
    def _reconstruction_forward(
        self, inputs, reconstruction, latent_mean, latent_logvar
    ):
        # TODO scale reconstruction to be same range as inputs

        recon_loss = self.reconstruction_loss_fn(
            reconstruction.reshape(-1),
            inputs.reshape(-1),
        )

        # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
        kl_divergence = (
            -0.5
            * torch.sum(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        ) * self.hparams.variational_beta

        out: Dict[str, torch.Tensor] = {
            "metrics/rec_loss": recon_loss,
            "metrics/kl_divergence_loss": kl_divergence,
            "reconstruction": reconstruction,
        }

        if self.hparams.get("norm_difference_loss", False):
            # penalize the reconstruction not having similar norm to the inputs
            channels_last = self.hparams.model.get("channels_last", True)
            input_norm = torch.linalg.norm(inputs, dim=2 if channels_last else 1)
            reconstruction_norm = torch.linalg.norm(
                reconstruction, dim=2 if channels_last else 1
            )
            norm_difference = (input_norm - reconstruction_norm).abs()
            norm_difference_loss = norm_difference.sum()
            assert norm_difference_loss.item() > 0
            out["metrics/norm_difference_loss"] = norm_difference_loss

        if self.hparams.get("diff_loss", False):
            # first-order differentiation of reconstruction must match that of output
            reconstruction_diff = reconstruction[:, 1:] - reconstruction[:, :-1]
            inputs_diff = inputs[:, 1:] - inputs[:, :-1]
            diff_loss = F.mse_loss(reconstruction_diff, inputs_diff, reduction="sum")
            out["metrics/diff_loss"] = diff_loss

        # IDEA: compute reconstruction STD and compare it to the inputs one.
        # their squared difference is a loss
        if self.hparams.get("std_difference_loss", False):
            std_difference = (reconstruction.std() - inputs.std()) ** 2
            out["metrics/std_difference_loss"] = std_difference

        return out
