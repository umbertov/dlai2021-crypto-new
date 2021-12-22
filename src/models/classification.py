from einops import rearrange
from torch import nn

from src.tcn import TemporalConvNet
from typing import Optional, Callable

from .model_utils import *
from .ffn import SimpleFeedForward


class Classifier(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        feature_dim = (
            feature_dim if feature_dim is not None else feature_extractor.out_dim
        )
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return {"classification_logits": self.classifier(features)}


class TcnClassifier(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_classes,
        sequence_length,
        num_channels,
        kernel_size,
        dropout=0.0,
        compression=1,
        channels_last=False,
        residual=False,
        clf_hidden_sizes=[20],
        activation=nn.LeakyReLU(),
        dilated_conv=True,
    ):
        super().__init__()
        TcnClass = TCNWrapper if channels_last else TemporalConvNet
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.activation = activation
        self.channels_last = channels_last

        self.encoder = TcnClass(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            downsample=compression,
            residual=residual,
            dilated=dilated_conv,
            activation=activation,
        )
        self.classifier = SimpleFeedForward(
            in_size=num_channels[-1],
            hidden_sizes=clf_hidden_sizes,
            out_size=self.num_classes,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        if not self.channels_last:
            encoded = encoded.transpose(-1, -2)
        clf_out = self.classifier(encoded)
        return {"classification_logits": clf_out}


class TcnSequenceClassifier(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_classes,
        sequence_length,
        num_channels,
        kernel_size,
        dropout=0.0,
        compression=1,
        channels_last=False,
        residual=False,
        clf_hidden_sizes=[20],
        activation=nn.LeakyReLU(),
        dilated_conv=True,
    ):
        super().__init__()
        TcnClass = TCNWrapper if channels_last else TemporalConvNet
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.activation = activation
        self.channels_last = channels_last

        self.encoder = TcnClass(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            downsample=compression,
            residual=residual,
            dilated=dilated_conv,
        )

        flattened_size = int(
            num_channels[-1] * (sequence_length // (compression ** (len(num_channels))))
        )
        self.flattened_size = flattened_size
        self.classifier = SimpleFeedForward(
            in_size=flattened_size,
            hidden_sizes=clf_hidden_sizes,
            out_size=self.num_classes,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = rearrange(encoded, "batch x y -> batch (x y)")
        clf_out = self.classifier(encoded)
        return {"classification_logits": clf_out}


class ClassifierWithAutoEncoder(nn.Module):
    def __init__(
        self,
        autoencoder,
        num_classes: int,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.num_classes = num_classes
        self.classifier = nn.Linear(autoencoder.hidden_sizes[-1], num_classes)

    def forward(self, x):
        x = self.feature_scaler(x)
        encoded, reconstruction = self.autoencoder(x)
        predictions = self.classifier(encoded)
        return {
            "classification_logits": predictions,
            "reconstruction": reconstruction,
        }
