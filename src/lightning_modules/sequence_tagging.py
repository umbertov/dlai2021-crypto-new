import pandas as pd
from einops import rearrange
import torch
from torch.nn import functional as F
import random

import pytorch_lightning as pl
import torchmetrics.functional as M

from src.common.plot_utils import confusion_matrix_fig, plot_categorical_tensor
from src.dyn_loss import DynamicWeightCrossEntropy
from src.lightning_modules.base import BaseTimeSeriesModule
from src.common.plot_utils import confusion_matrix_fig


from src.common.utils import compute_confusion_matrix, compute_classification_metrics


class TimeSeriesClassifier(BaseTimeSeriesModule):
    def _classification_forward(
        self, classification_logits=None, categorical_targets=None, **_
    ):
        if classification_logits is None or categorical_targets is None:
            return dict(classification_logits=classification_logits)
        original_classification_logits = classification_logits
        if classification_logits.size(1) == 1:  # seq. len of 1
            categorical_targets = categorical_targets[:, [-1], :]
        classification_logits = rearrange(classification_logits, "b l c -> (b l) c")
        classification_loss = self.classification_loss_fn(
            classification_logits,
            categorical_targets.view(-1),
        )
        out = {
            "metrics/clf_loss": classification_loss,
            "categorical_targets": categorical_targets,
            **compute_classification_metrics(
                classification_logits,
                categorical_targets.view(-1),
                num_classes=self.hparams.model.num_classes,
            ),
            "classification_logits": original_classification_logits,
        }
        return out

    def _classifier_epoch_end(self, step_outputs, mode: str):
        if self.classification_loss_fn is not None:
            confusion_matrix_plot = self._confusion_matrix_plot(
                step_outputs, self.hparams.model.num_classes
            )
            self.logger.experiment.log(
                {f"{mode}/confusion_matrix": confusion_matrix_plot}
            )
            if isinstance(self.logger, pl.loggers.WandbLogger):
                chunks_f1 = self._compute_f1_by_position(step_outputs, 4)
                chunks_f1["epoch"] = self.trainer.current_epoch
                self.logger.log_table(
                    f"{mode}/f1_by_pos",
                    dataframe=pd.DataFrame(
                        chunks_f1, index=[self.trainer.current_epoch]
                    ),
                )
            # categorical_plot = self._categorical_data_plot(step_outputs)
            # self.logger.experiment.log(
            #     {f"{mode}/categorical_data_plot": categorical_plot}
            # )
        if (
            isinstance(self.classification_loss_fn, DynamicWeightCrossEntropy)
            and "train" in mode
            and isinstance(self.logger, pl.loggers.WandbLogger)
        ):
            #### LOSS WEIGHTS
            print("LOGGING LOSS WEIGHTSSSSSSSSSSSSSSSSSSSS")
            print(self.classification_loss_fn.weight)
            self.logger.experiment.log(
                {
                    f"{mode}/loss_weights": self.classification_loss_fn.weight.cpu().numpy()
                }
            )

    def _confusion_matrix_plot(self, step_outputs, num_classes):
        assert step_outputs
        # Log confusion matrix for the training data
        confusion_matrix = (
            compute_confusion_matrix(
                torch.cat(
                    [
                        rearrange(step["classification_logits"], "b s c -> (b s) c")
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

    def _compute_f1_by_position(self, step_outputs, chunk_size=4):
        # [len(dataset), seqlen, classes]
        all_clf_logits = torch.cat(
            [s["classification_logits"] for s in step_outputs], dim=0
        )
        all_clf_targets = torch.cat(
            [s["categorical_targets"] for s in step_outputs], dim=0
        ).squeeze()
        _, seqlen, _ = all_clf_logits.shape
        chunk_size = seqlen // 4
        out = dict()
        for i in range(0, 4):
            start = i * chunk_size
            end = start + chunk_size
            logits = all_clf_logits[:, start:end]
            targets = all_clf_targets[:, start:end]
            out[f"chunk_{i}/f1"] = M.f1(
                rearrange(logits, "batch seq class -> (batch seq) class"),
                targets.reshape(-1),
                average="macro",
                num_classes=self.hparams.model.num_classes,
            ).item()
        return out

    def _categorical_data_plot(self, step_outputs):
        assert step_outputs
        random_step = random.choice(step_outputs)
        inputs, categorical_targets = (
            random_step["inputs"][0],
            random_step["categorical_targets"][0],
        )
        groundtruth_plot = plot_categorical_tensor(
            inputs.cpu().transpose(-1, -2), categorical_targets.cpu().transpose(-1, -2)
        )
        return groundtruth_plot
