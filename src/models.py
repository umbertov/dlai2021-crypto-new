from einops.einops import rearrange
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch.nn.init as I
from torch.nn import functional as F
import math

from typing import Callable, Optional
from einops.layers.torch import Rearrange


class LambdaLayer(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.forward = f


# stolen from huggingface transformers
def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def NonLinear(
    in_size: int,
    hidden_size: int,
    activation: nn.Module = nn.Sigmoid(),
    dropout: float = 0.0,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_size, hidden_size, bias=False),
        nn.Dropout(dropout),
        activation,
        Rearrange("batch seqlen channels -> batch channels seqlen"),
        nn.BatchNorm1d(hidden_size),
        Rearrange("batch channels seqlen -> batch seqlen channels"),
    )


def init_gru(cell, gain=1):
    cell.reset_parameters()

    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            I.orthogonal(hh[i : i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4 : l // 2].data.fill_(1.0)
        hh_b[l // 4 : l // 2].data.fill_(1.0)


class SinePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinePositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class CausalTransformer(nn.Module):
    def __init__(
        self,
        feature_size=256,
        num_layers=1,
        n_heads=4,
        dropout=0.1,
    ):
        super(CausalTransformer, self).__init__()
        self.feature_size = feature_size

        self.src_mask = None
        self.pos_encoder = SinePositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=n_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Identity()
        self.init_weights()

    def init_weights(self):
        if isinstance(self.decoder, nn.Linear):
            initrange = 0.1
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        """Upper triangular attention mask to enforce causal attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        in_size,
        transformer: CausalTransformer,
    ):
        super().__init__()
        self.in_size = in_size
        self.embedding = nn.Linear(in_size, transformer.feature_size, bias=False)
        self.transformer = transformer
        self.out_dim = transformer.feature_size

    def forward(self, x):
        embedded = self.embedding(x)
        out = self.transformer(embedded)
        return out


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
        flatten_input=True,
    ):
        super().__init__()
        if flatten_input:
            in_size *= window_length
        self.in_size = in_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.window_length = window_length
        self.out_dim = self.hidden_sizes[-1]
        self.flatten_input = flatten_input

        # list of tuples of adjacent layer sizes
        projection_sizes = list(zip([in_size] + hidden_sizes, hidden_sizes))

        self.net = nn.Sequential(
            *[
                NonLinear(s1, s2, activation=self.activation, dropout=self.dropout)
                for (s1, s2) in projection_sizes
            ]
        )

    def forward(self, x):
        if x.dim() == 3 and self.flatten_input:
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
        init_lstm(self.lstm)
        self.out_dim = self.window_length * self.hidden_size

    def forward(self, x):
        encoded_seq, _ = self.lstm(x)
        return encoded_seq
        # return rearrange(encoded_seq, "batch seqlen f -> batch (seqlen f)")


class AutoregressiveLstmModel(LstmModel):
    def forward(self, x, n_autoregression_steps=0):
        raise NotImplementedError


class LstmMLP(nn.Module):
    def __init__(self, lstm: LstmModel, mlp: SimpleFeedForward):
        super().__init__()
        self.lstm = lstm
        self.mlp = mlp
        self.model = nn.Sequential(lstm, mlp)
        self.in_size = self.lstm.in_size
        self.out_dim = self.mlp.out_dim

    def forward(self, x):
        return self.model(x)


class FeatureScaler(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.factors = nn.Parameter(torch.ones(in_size), requires_grad=True)

    def forward(self, x):
        return x * self.factors


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
        use_feature_scaler=False,
    ):
        super().__init__()
        feature_dim = (
            feature_dim if feature_dim is not None else feature_extractor.out_dim
        )
        self.feature_scaler = (
            FeatureScaler(feature_extractor.in_size)
            if use_feature_scaler
            else nn.Identity()
        )
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        x = self.feature_scaler(x)
        features = self.feature_extractor(x)
        return {"classification_logits": self.classifier(features)}


class ClassifierWithAutoEncoder(nn.Module):
    def __init__(
        self, autoencoder: AutoEncoder, n_classes: int, use_feature_scaler=False
    ):
        super().__init__()
        self.feature_scaler = (
            FeatureScaler(autoencoder.in_size) if use_feature_scaler else nn.Identity()
        )
        self.autoencoder = autoencoder
        self.n_classes = n_classes
        self.classifier = nn.Linear(autoencoder.hidden_sizes[-1], n_classes)

    def forward(self, x):
        x = self.feature_scaler(x)
        encoded, reconstruction = self.autoencoder(x)
        predictions = self.classifier(encoded)
        return {
            "classification_logits": predictions,
            "reconstruction": reconstruction,
        }


class Regressor(nn.Module):
    def __init__(
        self,
        feature_extractor: SimpleFeedForward,
        n_outputs: int,
        feature_dim: Optional[int] = None,
        use_feature_scaler=False,
    ):
        super().__init__()
        feature_dim = (
            feature_dim if feature_dim is not None else feature_extractor.out_dim
        )
        self.feature_scaler = (
            FeatureScaler(feature_extractor.in_size)
            if use_feature_scaler
            else nn.Identity()
        )
        self.feature_extractor = feature_extractor
        self.n_outputs = n_outputs
        self.feature_dim = feature_dim
        self.decoder = nn.Linear(feature_dim, n_outputs)

    def forward(self, x):
        x = self.feature_scaler(x)
        features = self.feature_extractor(x)
        return {"regression_output": self.decoder(features)}
