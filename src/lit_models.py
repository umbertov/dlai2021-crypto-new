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


def notnone(x):
    return x is not None


def regression_plot_fig(regression_outs, targets):
    regression_out = regression_outs[0]
    target = targets[0]
    assert regression_out.shape == target.shape
    length, num_channels = regression_out.shape

    if num_channels == len("ohlc"):
        return plot_multi_ohlcv(prediction=regression_out.cpu(), truth=target.cpu())

    else:
        fig = make_subplots(
            rows=num_channels,
            cols=1,
            subplot_titles=[f"Channel #{i}" for i in range(num_channels)],
        )
        for channel in range(num_channels):
            traces = plot_multi_lines(
                truth=target[:, channel].view(-1).cpu(),
                prediction=regression_out[:, channel].view(-1).cpu(),
            )
            for trace in traces:
                fig.add_trace(trace, row=channel + 1, col=1)
        return fig


# def time_discount_mse_loss(x, y, sequence_length, alpha=1.1):
#     """x,y : (Batch, Seqlen, Channels)"""
#     assert x.shape == y.shape
#     losses = F.mse_loss(x, y, reduction="none")
#
#     # exponentially rising multiplicative coefficients
#     _, seqlen, _ = x.shape
#     coefficients = alpha ** (torch.arange(seqlen, device=x.device) - seqlen + 1)
#     coefficients = coefficients[None, :, None]
#
#     return torch.mean(losses * coefficients)
#
#
# class TimeDiscountMseLoss(nn.Module):
#     def __init__(self, alpha=1.1):
#         super().__init__()
#         self.alpha = alpha
#
#     def forward(self, x, y):
#         sequence_length = x.size(1)
#         return time_discount_mse_loss(
#             x, y, sequence_length=sequence_length, alpha=self.alpha
#         )
#
#
# def ohlc_prod_mse_loss(x, y):
#     loss = F.mse_loss(x, y, reduction="none")
#     assert loss.dim() == 3
#     loss = loss.prod(dim=-1)
#     return loss.sum()
#
#
# def OhlcProdMseLoss():
#     return ohlc_prod_mse_loss


def compute_accuracy(logits, targets, num_classes):
    return M.accuracy(
        F.softmax(logits, dim=-1),
        targets,
        average="macro",
        num_classes=num_classes,
    )


def compute_classification_metrics(logits, targets, num_classes):
    pred_probabilities = F.softmax(logits, dim=-1)
    return {
        "metrics/accuracy": M.accuracy(
            pred_probabilities, targets, num_classes=num_classes, average="macro"
        ),
        "metrics/precision": M.precision(
            pred_probabilities, targets, num_classes=num_classes, average="macro"
        ),
        "metrics/recall": M.recall(
            pred_probabilities, targets, num_classes=num_classes, average="macro"
        ),
        "metrics/f1": M.f1(
            pred_probabilities, targets, num_classes=num_classes, average="macro"
        ),
    }


def compute_confusion_matrix(logits, targets, num_classes):
    return M.confusion_matrix(
        F.softmax(logits.detach(), dim=-1),
        targets.detach().view(-1),
        normalize="true",  # normalize over targets ('true') or predictions ('pred')
        num_classes=num_classes,
    )


def regression_baseline_score(
    loss_fn: nn.MSELoss, targets: torch.FloatTensor
) -> torch.Tensor:
    """Computes the regression loss for a model that predicts the same value for
    the future as in the present
    """
    return loss_fn(targets[:, 1:], targets[:, :-1])


class BaseTimeSeriesModule(pl.LightningModule):
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

        self.regression_loss_fn = hydra.utils.instantiate(
            self.hparams.regression_loss_fn,
            _recursive_=True,
        )

        self.classification_loss_fn = hydra.utils.instantiate(
            self.hparams.classification_loss_fn
        )
        self.reconstruction_loss_fn = hydra.utils.instantiate(
            self.hparams.reconstruction_loss_fn,
            _recursive_=True,
        )

    def forward(self, inputs: torch.Tensor, **kwargs):

        out = dict(inputs=inputs)

        model_out = self.model(inputs)

        out.update(model_out)

        results = list(
            filter(
                notnone,
                [
                    self._regression_forward(**model_out, **kwargs),
                    self._future_regression_forward(**model_out, **kwargs),
                    self._reconstruction_forward(**model_out, inputs=inputs),
                    self._classification_forward(**model_out, **kwargs),
                ],
            )
        )
        assert len(results) > 0
        assert not all(x is None for x in results)
        for d in results:
            out.update(d)

        losses = [
            value
            for key, value in out.items()
            if key.endswith("_loss") and not "baseline" in key
        ]
        if losses:
            out["loss"] = torch.stack(losses).sum()
            out["metrics/loss"] = out["loss"]
        return {k: v.detach() if k != "loss" else v for k, v in out.items()}

    def step(self, batch, batch_idx, mode):
        if isinstance(batch, dict):
            step_result = self.forward(**batch)
        elif isinstance(batch, list):
            step_result = self.forward(*batch)
        else:
            step_result = self.forward(batch)
        metrics = {
            f"{mode}/{key.split('/')[1]}": value
            for key, value in step_result.items()
            if key.startswith("metrics/")
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return step_result

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")

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
        self._classifier_epoch_end(step_outputs, mode="train")
        random_step = random.choice(step_outputs)
        self._regression_epoch_end(random_step, mode="train")
        self._reconstruction_epoch_end(random_step, mode="train")

    def validation_epoch_end(self, step_outputs):
        self._classifier_epoch_end(step_outputs, mode="val")
        random_step = random.choice(step_outputs)
        self._regression_epoch_end(random_step, mode="val")
        self._reconstruction_epoch_end(random_step, mode="val")

    #### EPOCH ENDS
    def _regression_epoch_end(self, step_outputs, mode: str):
        pass

    def _classifier_epoch_end(self, step_outputs, mode: str):
        pass

    def _reconstruction_epoch_end(self, random_step, mode: str):
        pass

    #### FORWARDS
    def _regression_forward(self, **_):
        return None

    def _future_regression_forward(self, **_):
        return None

    def _reconstruction_forward(self, **_):
        return None

    def _classification_forward(self, **_):
        return None

    #### MISC
    # def on_after_backward(self, *args, **kwargs):
    #     # nan params
    #     if any(p.isnan().any().item() for p in self.parameters()):
    #         import ipdb

    #         ipdb.set_trace()
    #     # nan gradients
    #     if any(
    #         p.grad.isnan().any().item() for p in self.parameters() if p.grad is not None
    #     ):
    #         import ipdb

    #         ipdb.set_trace()
    #     return super().on_after_backward(*args, **kwargs)


class TimeSeriesRegressor(BaseTimeSeriesModule):
    def _regression_forward(self, regression_output=None, continuous_targets=None, **_):
        if regression_output is None or continuous_targets is None:
            return None
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
        self, future_regression_output=None, future_continuous_targets=None, **_
    ):
        out = self._regression_forward(
            regression_output=future_regression_output,
            continuous_targets=future_continuous_targets,
        )
        return (
            {
                "metrics/future_reg_loss": out["metrics/reg_loss"],
                "metrics/future_baseline_reg_loss": out["metrics/reg_loss"],
                "future_continuous_targets": out["continuous_targets"],
            }
            if out is not None
            else None
        )

    def _regression_epoch_end(self, random_step, mode: str):
        if self.regression_loss_fn is not None:
            plot = regression_plot_fig(
                random_step["regression_output"], random_step["continuous_targets"]
            )
            self.logger.experiment.log({f"{mode}/prediction_plot": plot})
            if "forecast" in random_step.keys():
                plot = regression_plot_fig(
                    random_step["forecast"], random_step["future_continuous_targets"]
                )
                self.logger.experiment.log({f"{mode}/forecast_plot": plot})


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

        out: dict[str, torch.Tensor] = {
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


class TimeSeriesClassifier(BaseTimeSeriesModule):
    def _classification_forward(
        self, classification_logits=None, categorical_targets=None, **_
    ):
        if classification_logits is None or categorical_targets is None:
            return dict(classification_logits=classification_logits)
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
            **compute_classification_metrics(
                classification_logits,
                categorical_targets.view(-1),
                num_classes=self.hparams.model.num_classes,
            ),
            "classification_logits": classification_logits,
        }

    def _classifier_epoch_end(self, step_outputs, mode: str):
        if self.classification_loss_fn is not None:
            confusion_matrix_plot = self._confusion_matrix_plot(
                step_outputs, self.hparams.model.num_classes
            )
            self.logger.experiment.log(
                {f"{mode}/confusion_matrix": confusion_matrix_plot}
            )

    def _confusion_matrix_plot(self, step_outputs, num_classes):
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
                num_classes=num_classes,
            )
            .cpu()
            .numpy()
        )
        plot = confusion_matrix_fig(
            confusion_matrix,
            labels=list(range(num_classes)),
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
        example_in_tensor[:, :, :], 0, cfg.dataset_conf.num_classes
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
