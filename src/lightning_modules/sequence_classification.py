from random import random
import torch
import pytorch_lightning as pl

from src.dyn_loss import DynamicWeightCrossEntropy
from src.common.plot_utils import confusion_matrix_fig, plot_categorical_tensor
from src.common.utils import compute_confusion_matrix, compute_classification_metrics

from .base import BaseTimeSeriesModule


class TimeSeriesSequenceClassifier(BaseTimeSeriesModule):
    def _classification_forward(
        self, classification_logits=None, categorical_targets=None, **_
    ):
        # classification_logits : [ Batch, N_Classes ]
        # categorical_targets : [ Batch, SeqLen]
        if classification_logits is None or categorical_targets is None:
            return dict(classification_logits=classification_logits)
        original_classification_logits = classification_logits

        # we need to get the last element of categorical_targets as the label for entire sequence
        categorical_targets = categorical_targets[..., -1]
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
            "classification_logits": original_classification_logits,
        }

    def _classifier_epoch_end(self, step_outputs, mode: str):
        if self.classification_loss_fn is not None:
            ### CONFUSION MATRIX
            confusion_matrix_plot = self._confusion_matrix_plot(
                step_outputs, self.hparams.model.num_classes
            )
            self.logger.experiment.log(
                {f"{mode}/confusion_matrix": confusion_matrix_plot}
            )

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
