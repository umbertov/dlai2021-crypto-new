from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
from torch import nn
from torch.nn import functional as F
import wandb
from plotly.subplots import make_subplots
import random

import pytorch_lightning as pl
import torchmetrics.functional as M
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from einops import rearrange
from src.common.plot_utils import (
    confusion_matrix_fig,
    plot_multi_lines,
    plot_multi_ohlcv,
)

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


def regression_baseline_score(
    loss_fn: nn.MSELoss, targets: torch.FloatTensor
) -> torch.Tensor:
    """Computes the regression loss for a model that predicts the same value for
    the future as in the present
    """
    return loss_fn(targets[:, 1:], targets[:, :-1])


from vector_quantize_pytorch import VectorQuantize


class CandlestickVqVAE(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        self.quantizer = hydra.utils.instantiate(
            self.hparams.quantizer, _recursive_=True, _convert_="partial"
        )

    def forward(self, inputs, *_, **__):
        quantized, indices, commitment_loss = self.quantizer(inputs)
        loss = F.mse_loss(inputs, quantized)
        return {"reconstruction": quantized, "loss": loss}


class TimeSeriesModule(pl.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        if "quantizer" in kwargs:
            self.quantizer = hydra.utils.instantiate(
                self.hparams.quantizer, _recursive_=True, _convert_="partial"
            )
        else:
            self.quantizer = None

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

    def _regression_forward(self, regression_output, continuous_targets):
        regression_loss = self.regression_loss_fn(regression_output, continuous_targets)
        baseline_loss = regression_baseline_score(
            self.regression_loss_fn, continuous_targets
        )
        return {
            "metrics/reg_loss": regression_loss,
            "metrics/baseline_reg_loss": baseline_loss,
            "continuous_targets": continuous_targets,
        }

    def _future_regression_forward(
        self, future_regression_output, future_continuous_targets
    ):
        out = self._regression_forward(
            future_regression_output, future_continuous_targets
        )
        return {
            "metrics/future_reg_loss": out["metrics/reg_loss"],
            "metrics/future_baseline_reg_loss": out["metrics/reg_loss"],
            "future_continuous_targets": out["continuous_targets"],
        }

    def _reconstruction_forward(self, reconstruction, inputs):
        reconstruction_loss = (
            self.hparams.reconstruction_loss_weight  # type: ignore
            * self.reconstruction_loss_fn(reconstruction.view(-1), inputs.view(-1))
        )
        return {
            "metrics/rec_loss": reconstruction_loss,
            "reconstruction": reconstruction,
        }

    def _classification_forward(self, classification_logits, categorical_targets):
        if classification_logits.size(1) == 1:  # seq. len of 1
            categorical_targets = categorical_targets[:, [-1], :]
        classification_logits = rearrange(classification_logits, "b l c -> (b l) c")
        classification_loss = self.classification_loss_fn(
            classification_logits,
            categorical_targets.view(-1),
        )
        return {
            "metrics/clf_loss": classification_loss,
            "categorical_targets": categorical_targets,
            "metrics/accuracy": compute_accuracy(
                classification_logits,
                categorical_targets.view(-1),
                n_classes=self.hparams.model.n_classes,
            ),
            "classification_logits": classification_logits,
        }

    def forward(
        self,
        inputs: torch.Tensor,
        continuous_targets: torch.Tensor = None,
        categorical_targets: torch.Tensor = None,
        future_continuous_targets: torch.Tensor = None,
    ):

        out = dict(inputs=inputs)

        if self.quantizer is not None:
            inputs_embedded, indices, commitment_loss = self.quantizer(inputs)
        else:
            inputs_embedded = inputs

        model_out = self.model(inputs_embedded)
        classification_logits = model_out.get("classification_logits", None)
        regression_output = model_out.get("regression_output", None)
        reconstruction = model_out.get("reconstruction", None)

        if reconstruction is None and self.quantizer is not None:
            reconstruction = inputs_embedded

        out.update(model_out)

        if regression_output is not None and continuous_targets is not None:
            out.update(self._regression_forward(regression_output, continuous_targets))
            if future_continuous_targets is not None:
                out.update(
                    self._future_regression_forward(
                        model_out["future_regression_output"], future_continuous_targets
                    )
                )

        if reconstruction is not None:
            out.update(self._reconstruction_forward(reconstruction, inputs))

        if classification_logits is not None and categorical_targets is not None:
            out.update(
                self._classification_forward(classification_logits, categorical_targets)
            )

        losses = [
            value
            for key, value in out.items()
            if key.endswith("_loss") and not "baseline" in key
        ]
        if losses:
            out["loss"] = torch.stack(losses).sum() / len(losses)
            out["metrics/loss"] = out["loss"]
        return {k: v.detach() if k != "loss" else v for k, v in out.items()}

    def training_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        metrics = {
            f"train/{key.split('/')[1]}": value
            for key, value in step_result.items()
            if key.startswith("metrics/")
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return step_result

    def validation_step(self, batch, batch_idx):
        step_result = self.forward(*batch)
        metrics = {
            f"val/{key.split('/')[1]}": value
            for key, value in step_result.items()
            if key.startswith("metrics/")
        }
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
        scheduler = {
            "scheduler": hydra.utils.instantiate(
                self.hparams.optim.lr_scheduler, optimizer=opt
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [scheduler]

    def training_epoch_end(self, step_outputs):
        random_step = random.choice(step_outputs)
        if self.classification_loss_fn is not None:
            confusion_matrix_plot = self._confusion_matrix_plot(step_outputs)
            self.logger.experiment.log(
                {"train/confusion_matrix": confusion_matrix_plot}
            )
        if self.regression_loss_fn is not None:
            plot = self._regression_plot_fig(
                random_step["regression_output"], random_step["continuous_targets"]
            )
            self.logger.experiment.log({"train/prediction_plot": plot})
        if self.reconstruction_loss_fn is not None:
            plot = self._regression_plot_fig(
                random_step["reconstruction"], random_step["inputs"]
            )
            self.logger.experiment.log({"train/reconstruction_plot": plot})

    def validation_epoch_end(self, step_outputs):
        random_step = random.choice(step_outputs)
        if self.classification_loss_fn is not None:
            confusion_matrix_plot = self._confusion_matrix_plot(step_outputs)
            self.logger.experiment.log({"val/confusion_matrix": confusion_matrix_plot})
        if self.regression_loss_fn is not None:
            plot = self._regression_plot_fig(
                random_step["regression_output"], random_step["continuous_targets"]
            )
            self.logger.experiment.log({"val/prediction_plot": plot})
        if self.reconstruction_loss_fn is not None:
            plot = self._regression_plot_fig(
                random_step["reconstruction"], random_step["inputs"]
            )
            self.logger.experiment.log({"val/reconstruction_plot": plot})

    def _regression_plot_fig(self, regression_outs, targets):
        regression_out = regression_outs[0]
        target = targets[0]
        assert regression_out.shape == target.shape
        length, n_channels = regression_out.shape

        if n_channels == len("ohlc"):
            return plot_multi_ohlcv(prediction=regression_out.cpu(), truth=target.cpu())

        else:
            fig = make_subplots(
                rows=n_channels,
                cols=1,
                subplot_titles=[f"Channel #{i}" for i in range(n_channels)],
            )
            for channel in range(n_channels):
                traces = plot_multi_lines(
                    truth=target[:, channel].view(-1).cpu(),
                    prediction=regression_out[:, channel].view(-1).cpu(),
                )
                for trace in traces:
                    fig.add_trace(trace, row=channel + 1, col=1)
            return fig

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
