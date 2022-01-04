from einops.einops import rearrange
from torch import nn
import torch.nn.init as I

from .model_utils import compute_forecast
from .ffn import SimpleFeedForward


def init_gru(cell, gain=1):
    cell.reset_parameters()

    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            I.orthogonal_(hh[i : i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4 : l // 2].data.fill_(1.0)
        hh_b[l // 4 : l // 2].data.fill_(1.0)


class LstmModel(nn.Module):
    forecast = compute_forecast

    def __init__(
        self,
        in_size,
        hidden_size,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
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
        init_lstm(self.lstm)
        self.out_dim = self.hidden_size

    def forward(self, x):
        encoded_seq, _ = self.lstm(x)
        return encoded_seq


class AutoregressiveLstmModel(LstmModel):
    def forward(self, x, n_autoregression_steps=0):
        raise NotImplementedError


class LstmMLP(nn.Module):
    forecast = compute_forecast

    def __init__(self, lstm: LstmModel, mlp: SimpleFeedForward):
        super().__init__()
        self.lstm = lstm
        self.mlp = mlp
        self.model = nn.Sequential(lstm, mlp)
        self.in_size = self.lstm.in_size
        self.out_dim = self.mlp.out_dim

    def forward(self, x):
        return self.model(x)
