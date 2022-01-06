from typing import Union, Sequence, Any, Tuple
from torch.optim import Optimizer
import torch
import random

import pytorch_lightning as pl

import hydra, omegaconf

from src.common.utils import notnone, PROJECT_ROOT


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
            step_result = self.forward(**batch, mode=mode)
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
        opt: Optimizer = hydra.utils.instantiate(
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

    def on_after_backward(self, *args, **kwargs):
        if not all(
            p.grad.isfinite().all().item()
            for p in self.parameters()
            if p.grad is not None
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
    if cfg.dataset_conf.channels_last == False:
        example_in_tensor = example_in_tensor.transpose(-1, -2).contiguous()
    example_categorical_tensor = torch.randint(
        0, cfg.dataset_conf.n_classes, (2, cfg.dataset_conf.window_length)
    ).long()
    example_continuous_tensor = torch.randn_like(example_in_tensor)

    example_output = model(
        example_in_tensor,
        continuous_targets=None,
        categorical_targets=example_categorical_tensor,
    )
    return dict(model=model, out=example_output, cfg=cfg)


if __name__ == "__main__":
    model = None
    main_res = main()
