from einops.einops import rearrange
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


class DictEmbedder(nn.Module):
    def __init__(self, key2hiddensize: dict[str, tuple[int, int]]):
        self.k2h = key2hiddensize
        self.modules_dict = dict()
        for key, (in_size, hidden_size) in key2hiddensize.items():
            self.modules_dict[key] = NonLinear(in_size=in_size, hidden_size=hidden_size)
        self.modules = nn.ModuleList(list(self.modules_dict.items()))

    def forward(self, **keys2tensors) -> dict[str, torch.Tensor]:
        if not all(k in self.modules_dict.keys() for k in keys2tensors):
            raise KeyError
        res = {
            key: self.modules_dict[key](input_tensor)
            for key, input_tensor in keys2tensors.items()
        }
        return res


class SimpleFeedForward(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        activation: nn.Module = nn.Sigmoid(),
        dropout: float = 0.0,
        window_length: int = 1,
    ):
        super().__init__()
        in_size *= window_length
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.window_length = window_length
        self.out_dim = self.hidden_sizes[-1]

        # list of tuples of adjacent layer sizes
        projection_sizes = list(zip([in_size] + hidden_sizes, hidden_sizes))

        self.net = nn.Sequential(
            *[
                NonLinear(s1, s2, activation=self.activation, dropout=self.dropout)
                for (s1, s2) in projection_sizes
            ]
        )

    def forward(self, x):
        if x.dim() == 3:
            x = rearrange(
                x, "batch window features -> batch (window features)"
            ).unsqueeze(1)
        return self.net(x)


class LstmModel(nn.Module):
    def __init__(self, in_size, hidden_size, window_length, num_layers, dropout=0.0):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.window_length = window_length
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.out_dim = self.window_length * self.hidden_size

    def forward(self, x):
        encoded_seq, _ = self.lstm(x)
        return encoded_seq
        # return rearrange(encoded_seq, "batch seqlen f -> batch (seqlen f)")


class LstmMLP(nn.Module):
    def __init__(self, lstm: LstmModel, mlp: SimpleFeedForward):
        super().__init__()
        self.lstm = lstm
        self.mlp = mlp
        self.model = nn.Sequential(lstm, mlp)
        self.out_dim = self.mlp.out_dim

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        activation: nn.Module = nn.Sigmoid(),
        dropout: float = 0.0,
        window_length: int = 1,
    ):
        super().__init__()
        in_size *= window_length
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.window_length = window_length

        self.encoder = SimpleFeedForward(
            in_size,
            hidden_sizes,
            activation,
            dropout=dropout,
        )
        decoder_hidden_sizes = list(reversed(hidden_sizes)) + [in_size]
        self.decoder = SimpleFeedForward(
            hidden_sizes[-1],
            decoder_hidden_sizes,
            activation,
            dropout=dropout,
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
            feature_dim if feature_dim is not None else feature_extractor.out_dim
        )
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return {"classification_logits": self.classifier(features)}


class ClassifierWithAutoEncoder(nn.Module):
    def __init__(self, autoencoder: AutoEncoder, n_classes: int):
        super().__init__()
        self.autoencoder = autoencoder
        self.n_classes = n_classes
        self.classifier = nn.Linear(autoencoder.hidden_sizes[-1], n_classes)

    def forward(self, x):
        encoded, reconstruction = self.autoencoder(x)
        predictions = self.classifier(encoded)
        return {
            "classification_logits": predictions,
            "reconstruction": reconstruction,
        }
