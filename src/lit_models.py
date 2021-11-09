from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
import torchmetrics.functional as M
from einops import rearrange

from src.models import ClassifierWithAutoEncoder
from src.common.utils import PROJECT_ROOT

import hydra
import omegaconf


def compute_accuracy(logits, targets):
    return M.accuracy(
        F.softmax(logits, dim=-1),
        targets,
        average="weighted",
        num_classes=3,
    )


class MLPClassifierModel(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.model = hydra.utils.instantiate(
            self.hparams.model, _recursive_=True, _convert_="partial"
        )

        self.classification_loss_fn = hydra.utils.instantiate(
            self.hparams.classification_loss_fn
        )

    def forward(
        self,
        inputs: torch.Tensor,
        continuous_targets: torch.Tensor = None,
        categorical_targets: torch.Tensor = None,
    ):
        prediction_logits = self.model(inputs)

        out = {
            "logits": prediction_logits,
        }

        if categorical_targets is not None:
            classification_loss = self.classification_loss_fn(
                prediction_logits, categorical_targets.view(-1)
            )
            out["clf_loss"] = classification_loss
            out["categorical_targets"] = categorical_targets
            out["loss"] = classification_loss
            out["accuracy"] = compute_accuracy(prediction_logits, categorical_targets)

        return out

    def training_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        self.log_dict(
            {
                "train/clf_loss": step_result["clf_loss"],
                "train/loss": step_result["loss"],
                "train/accuracy": step_result["accuracy"],
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
                "val/loss": step_result["loss"],
                "val/accuracy": step_result["accuracy"],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_result

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if self.hparams.optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler, optimizer=opt
            )
            return [opt], [scheduler]

        return [opt]


class AutoEncoderModel(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.model = hydra.utils.instantiate(
            self.hparams.model, _recursive_=True, _convert_="partial"
        )

        self.classification_loss_fn = hydra.utils.instantiate(
            self.hparams.classification_loss_fn
        )
        self.reconstruction_loss_fn = hydra.utils.instantiate(
            self.hparams.reconstruction_loss_fn
        )

    def forward(
        self,
        inputs: torch.Tensor,
        continuous_targets: torch.Tensor = None,
        categorical_targets: torch.Tensor = None,
    ):
        prediction_logits, reconstruction = self.model(inputs)

        out = {
            "logits": prediction_logits,
            "reconstruction": reconstruction,
        }

        if continuous_targets is not None:
            reconstruction_loss = (
                self.hparams.reconstruction_loss_weight  # type: ignore
                * self.reconstruction_loss_fn(reconstruction.view(-1), inputs.view(-1))
            )
            out["rec_loss"] = reconstruction_loss
            out["continuous_targets"] = continuous_targets
            out["loss"] = reconstruction_loss

        if categorical_targets is not None:
            classification_loss = self.classification_loss_fn(
                prediction_logits, categorical_targets.view(-1)
            )
            out["clf_loss"] = classification_loss
            out["categorical_targets"] = categorical_targets
            out["loss"] = out.get("rec_loss", 0) + classification_loss
            out["accuracy"] = compute_accuracy(prediction_logits, categorical_targets)

        return out

    def training_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        self.log_dict(
            {
                "train/clf_loss": step_result["clf_loss"],
                "train/rec_loss": step_result["rec_loss"],
                "train/loss": step_result["loss"],
                "train/accuracy": step_result["accuracy"],
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
                "val/loss": step_result["loss"],
                "val/accuracy": step_result["accuracy"],
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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    global model
    model = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        # logging=cfg.logging,
        _recursive_=False,
    )
    in_size = len(cfg.dataset_conf.input_columns)
    example_tensor = torch.randn(2, in_size)
    example_output = model(example_tensor)


if __name__ == "__main__":
    model = None
    main()
