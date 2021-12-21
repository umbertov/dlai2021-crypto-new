from torch import nn

from .model_utils import *


class SimpleFeedForward(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: List[int],
        out_size=None,
        activation: nn.Module = nn.Sigmoid(),
        dropout: float = 0.0,
        window_length: int = 1,
        flatten_input=False,
        channels_last=True,
    ):
        """
        if out_size is not None, feed forward neural network as follows:
            {
                - [
                    - Linear layer
                    - Dropout
                    - Activation

                    for each dimension in hidden_sizes
                ]
                - Optionally: Linear(hidden_sizes[-1] -> out_size)
                    (depending on whether out_size is not None)
            }

        """
        assert len(hidden_sizes) > 0 or out_size is not None
        super().__init__()
        if flatten_input:
            in_size *= window_length
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.window_length = window_length
        self.out_dim = self.hidden_sizes[-1] if len(self.hidden_sizes) > 0 else out_size
        self.flatten_input = flatten_input
        self.channels_last = channels_last

        # list of tuples of adjacent layer sizes
        sizes = [in_size] + hidden_sizes
        projection_sizes = list(zip(sizes, sizes[1:]))

        final_layer = (
            nn.Linear(sizes[-1], out_size) if out_size is not None else nn.Identity()
        )

        self.net = nn.Sequential(
            *[
                NonLinear(
                    s1,
                    s2,
                    activation=self.activation,
                    dropout=self.dropout,
                    channels_last=channels_last,
                )
                for (s1, s2) in projection_sizes
            ],
            final_layer,
        )

    def forward(self, x):
        if x.dim() == 3 and self.flatten_input:
            x = rearrange(
                x, "batch window features -> batch (window features)"
            ).unsqueeze(1)
        return self.net(x)
