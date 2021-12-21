import torch
from torch import nn
from src.common.plot_utils import plot_multi_lines, plot_multi_ohlcv


from src.lightning_modules.base import BaseTimeSeriesModule


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


def regression_baseline_score(
    loss_fn: nn.MSELoss, targets: torch.FloatTensor
) -> torch.Tensor:
    """Computes the regression loss for a model that predicts the same value for
    the future as in the present
    """
    return loss_fn(targets[:, 1:], targets[:, :-1])


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
