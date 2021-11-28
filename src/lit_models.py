from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
from torch import nn
from torch.nn import functional as F
import wandb

import pytorch_lightning as pl
import torchmetrics.functional as M
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from einops import rearrange
from common.plot_utils import confusion_matrix_fig
from src.common.plot_utils import confusion_matrix_table

from src.models import ClassifierWithAutoEncoder
from src.common.utils import PROJECT_ROOT

import hydra
import omegaconf


def compute_accuracy(logits, targets, n_classes):
    return M.accuracy(
        F.softmax(logits, dim=-1),
        targets,
        average="macro",
        num_classes=n_classes,
    )


def compute_confusion_matrix(logits, targets, n_classes):
    return M.confusion_matrix(
        F.softmax(logits.detach(), dim=-1),
        targets.detach().view(-1),
        normalize="true",  # normalize over targets ('true') or predictions ('pred')
        num_classes=n_classes,
    )


class TimeSeriesModule(pl.LightningModule):
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
        self.flatten_input = kwargs.get("flatten_input", True)

        self.classification_loss_fn = hydra.utils.instantiate(
            self.hparams.classification_loss_fn
        )
        self.reconstruction_loss_fn = hydra.utils.instantiate(
            self.hparams.reconstruction_loss_fn,
            _recursive_=True,
        )
        self.regression_loss_fn = hydra.utils.instantiate(
            self.hparams.regression_loss_fn,
            _recursive_=True,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        continuous_targets: torch.Tensor = None,
        categorical_targets: torch.Tensor = None,
    ):
        if any(p.isnan().any().item() for p in self.parameters()):
            import ipdb

            ipdb.set_trace()
        if inputs.isnan().any().item():
            return None

        model_out = self.model(inputs)
        classification_logits = model_out.get("classification_logits", None)
        regression_output = model_out.get("regression_output", None)
        reconstruction = model_out.get("reconstruction", None)

        out = dict()

        if regression_output is not None:
            regression_loss = self.regression_loss_fn(
                regression_output.view(-1), continuous_targets.view(-1)
            )
            out.update({"reg_loss": regression_loss})

        if reconstruction is not None:
            reconstruction_loss = (
                self.hparams.reconstruction_loss_weight  # type: ignore
                * self.reconstruction_loss_fn(reconstruction.view(-1), inputs.view(-1))
            )
            out.update(
                {
                    "rec_loss": reconstruction_loss,
                    "reconstruction": reconstruction,
                }
            )

        if classification_logits is not None and categorical_targets is not None:
            if classification_logits.size(1) == 1:  # seq. len of 1
                categorical_targets = categorical_targets[:, [-1], :]
            classification_logits = rearrange(classification_logits, "b l c -> (b l) c")
            classification_loss = self.classification_loss_fn(
                classification_logits,
                categorical_targets.view(-1),
            )
            out.update(
                {
                    "clf_loss": classification_loss,
                    "categorical_targets": categorical_targets,
                    "accuracy": compute_accuracy(
                        classification_logits,
                        categorical_targets.view(-1),
                        n_classes=self.hparams.model.n_classes,
                    ),
                    "classification_logits": classification_logits,
                }
            )

        out["loss"] = torch.stack(
            [value for key, value in out.items() if key.endswith("_loss")]
        ).sum()
        return {k: v.detach() if k != "loss" else v for k, v in out.items()}

    def on_after_backward(self, *args, **kwargs):
        # nan params
        if any(p.isnan().any().item() for p in self.parameters()):
            import ipdb

            ipdb.set_trace()
        # nan gradients
        if any(
            p.grad.isnan().any().item() for p in self.parameters() if p.grad is not None
        ):
            import ipdb

            ipdb.set_trace()
        return super().on_after_backward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        if step_result is None or step_result["loss"].isnan().item():
            self.zero_grad()
            return None
        metrics = {
            "train/loss": step_result["loss"],
        }
        if step_result.get("rec_loss", False):
            metrics["train/rec_loss"] = step_result["rec_loss"]
        if step_result.get("clf_loss", False):
            metrics["train/clf_loss"] = step_result["clf_loss"]
            metrics["train/accuracy_macro"] = step_result["accuracy"]

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return step_result

    def validation_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        if step_result is None or step_result["loss"].isnan().item():
            self.zero_grad()
            return None
        metrics = {
            "val/loss": step_result["loss"],
        }
        if step_result.get("rec_loss", False):
            metrics["val/rec_loss"] = step_result["rec_loss"]
        if step_result.get("clf_loss", False):
            metrics["val/clf_loss"] = step_result["clf_loss"]
            metrics["val/accuracy_macro"] = step_result["accuracy"]

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
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

    def training_epoch_end(self, step_outputs):
        if self.classification_loss_fn is not None:
            confusion_matrix_plot = self._confusion_matrix_plot(step_outputs)
            self.logger.experiment.log(
                {"train/confusion_matrix": confusion_matrix_plot}
            )

    def validation_epoch_end(self, step_outputs):
        if self.classification_loss_fn is not None:
            confusion_matrix_plot = self._confusion_matrix_plot(step_outputs)
            self.logger.experiment.log({"val/confusion_matrix": confusion_matrix_plot})

    def _confusion_matrix_plot(self, step_outputs):
        assert step_outputs
        # Log confusion matrix for the training data
        confusion_matrix = (
            compute_confusion_matrix(
                torch.cat(
                    [
                        step["classification_logits"]
                        for step in step_outputs
                        if step is not None
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        step["categorical_targets"]
                        for step in step_outputs
                        if step is not None
                    ],
                    dim=0,
                ),
                n_classes=self.hparams.model.n_classes,
            )
            .cpu()
            .numpy()
        )
        plot = confusion_matrix_fig(
            confusion_matrix,
            labels=list(range(self.hparams.model.n_classes)),
        )
        return plot


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
    example_in_tensor = torch.randn(2, cfg.dataset_conf.window_length, in_size)
    example_categorical_tensor = torch.randint_like(
        example_in_tensor[:, :, :], 0, cfg.dataset_conf.n_classes
    ).long()
    example_continuous_tensor = torch.randn_like(example_in_tensor)

    example_output = model(
        example_in_tensor,
        continuous_targets=example_continuous_tensor,
        categorical_targets=example_categorical_tensor,
    )


if __name__ == "__main__":
    model = None
    main()
