import torch
from torch import nn
from torch.nn import functional as F


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

        # list of tuples of adjacent layer sizes
        projection_sizes = list(zip([in_size] + hidden_sizes, hidden_sizes))

        self.encoder = nn.ModuleList(
            [
                NonLinear(s1, s2, activation=self.activation)
                for (s1, s2) in projection_sizes
            ]
        )

        self.decoder = nn.ModuleList(
            [
                NonLinear(s2, s1, activation=self.activation)
                for (s1, s2) in reversed(projection_sizes)
            ]
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        reconstruction = self.decoder(x)
        return encoded, reconstruction


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
