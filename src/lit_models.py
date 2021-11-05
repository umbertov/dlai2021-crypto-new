from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from einops import rearrange

from src.models import ClassifierWithAutoEncoder

import hydra


class AutoEncoderModel(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.model = hydra.utils.instantiate(self.hparams.model, recursive=True)

        self.classification_loss_fn = hydra.utils.instantiate(
            self.hparams.classification_loss_fn
        )
        self.reconstruction_loss_fn = hydra.utils.instantiate(
            self.hparams.classification_loss_fn
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        prediction_logits, reconstruction = self.model(inputs)

        if targets is not None:
            return {"logits": prediction_logits}

        reconstruction_loss = self.reconstruction_loss_fn(
            reconstruction.view(-1), inputs.view(-1)
        )
        classification_loss = self.classification_loss_fn(
            prediction_logits, targets.view(-1)
        )

        return {
            "clf_loss": classification_loss,
            "rec_loss": reconstruction_loss,
            "logits": prediction_logits,
            "targets": targets,
        }

    def training_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        self.log_dict(
            {
                "train/clf_loss": step_result["clf_loss"],
                "train/rec_loss": step_result["rec_loss"],
                "loss": step_result["rec_loss"] + step_result["clf_loss"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_result

    def validation_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        self.log_dict(
            {
                "val/clf_loss": step_result["clf_loss"],
                "val/rec_loss": step_result["rec_loss"],
                "loss": step_result["rec_loss"] + step_result["clf_loss"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_result

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]
