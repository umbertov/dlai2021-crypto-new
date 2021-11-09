import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional


def NonLinear(
    in_size: int,
    hidden_size: int,
    activation: nn.Module = nn.Sigmoid(),
    dropout: float = 0.0,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.Dropout(dropout),
        activation,
    )


class SimpleFeedForward(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        activation: nn.Module = nn.Sigmoid(),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout

        # list of tuples of adjacent layer sizes
        projection_sizes = list(zip([in_size] + hidden_sizes, hidden_sizes))

        self.net = nn.Sequential(
            *[
                NonLinear(s1, s2, activation=self.activation, dropout=self.dropout)
                for (s1, s2) in projection_sizes
            ]
        )

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        activation: nn.Module = nn.Sigmoid(),
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.encoder = SimpleFeedForward(in_size, hidden_sizes, activation)
        decoder_hidden_sizes = list(reversed(hidden_sizes)) + [in_size]
        self.decoder = SimpleFeedForward(
            hidden_sizes[-1], decoder_hidden_sizes, activation
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return encoded, reconstruction


class Classifier(nn.Module):
    def __init__(
        self,
        feature_extractor: SimpleFeedForward,
        n_classes: int,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        feature_dim = (
            feature_dim
            if feature_dim is not None
            else feature_extractor.hidden_sizes[-1]
        )
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class ClassifierWithAutoEncoder(nn.Module):
    def __init__(self, autoencoder: AutoEncoder, n_classes: int):
        super().__init__()
        self.autoencoder = autoencoder
        self.n_classes = n_classes
        self.classifier = nn.Linear(autoencoder.hidden_sizes[-1], n_classes)

    def forward(self, x):
        encoded, reconstruction = self.autoencoder(x)
        predictions = self.classifier(encoded)
        return predictions, reconstruction
