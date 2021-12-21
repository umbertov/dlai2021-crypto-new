from torch import nn
from .ffn import SimpleFeedForward

from .model_utils import *


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x) -> dict:
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return {"encoded": encoded, "reconstruction": reconstruction}


class FeedForwardAutoEncoder(AutoEncoder):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        activation: nn.Module = nn.Sigmoid(),
        dropout: float = 0.0,
        window_length: int = 1,
    ):
        in_size *= window_length

        encoder = SimpleFeedForward(
            in_size,
            hidden_sizes,
            activation,
            dropout=dropout,
        )
        decoder_hidden_sizes = list(reversed(hidden_sizes)) + [in_size]
        decoder = SimpleFeedForward(
            hidden_sizes[-1],
            decoder_hidden_sizes,
            activation,
            dropout=dropout,
        )
        super().__init__(encoder, decoder)

        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.window_length = window_length
