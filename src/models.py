from einops.einops import rearrange, repeat
import torch
from torch.nn.utils.weight_norm import weight_norm
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch.nn.init as I
from torch.nn import functional as F
import math
from functools import reduce
from operator import mul
from typing import Callable, Optional
from einops.layers.torch import Rearrange

from src.tcn import TemporalConvNet


class LambdaLayer(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.forward = f


def conv_output_shape(input_length, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor

    h = floor(
        ((input_length + (padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )
    return h


def calc_tcn_outs(input_length, num_channels, kernel_size=2, stride=1):
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
        dilation_size = 2 ** i
        in_channels = 0 if i == 0 else num_channels[i - 1]
        out_channels = num_channels[i]
        layers += [
            {
                "out_length": conv_output_shape(
                    input_length,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation_size,
                    padding=dilation_size * (kernel_size - 1),
                )
                - 1,
                "in_channels": in_channels,
                "out_channels": out_channels,
            }
        ]
        input_length = input_length / 2
    return layers


def log_scaled_mse(
    x,
    y,
    reduction,
    alpha=5,
    eps=1e-10,
):
    mse = F.mse_loss(x, y, reduction=reduction)
    x_log = torch.log(x + eps)
    y_log = torch.log(y + eps)
    assert x_log.isfinite().all()
    assert y_log.isfinite().all()
    log_mse = F.mse_loss(x, y, reduction=reduction)
    return mse + alpha * log_mse


def LogScaledMSELoss(reduction="mean"):
    def f(*args, **kwargs):
        kwargs["reduction"] = reduction
        return log_scaled_mse(
            *args,
            **kwargs,
        )

    return f


def tuple_to_index(tup, names_to_classes):
    result = 0
    prev_classes = 1
    for arg, n_classes in zip(tup, names_to_classes.values()):
        result += arg * prev_classes
        prev_classes = n_classes
    return result


def dict_to_index(dic, names_to_classes):
    result = 0
    prev_classes = 1
    for name, n_classes in names_to_classes.items():
        arg = dic[name]
        result += arg * prev_classes
        prev_classes = n_classes
    return result


class CartesianProdEmbedding(nn.Module):
    def __init__(self, hidden_size, **names_to_n_classes):
        super().__init__()
        self.names_to_classes = names_to_n_classes
        self.n_variables = len(names_to_n_classes)
        self.total_n_embeddings = reduce(mul, names_to_n_classes.items())
        self.embedding = nn.Embedding(self.total_n_embeddings, hidden_size)

    def forward(self, *args, **kwargs):
        if args:
            embedding_idx = tuple_to_index(args, self.names_to_classes)
        elif kwargs:
            embedding_idx = dict_to_index(kwargs, self.names_to_classes)
        else:
            raise ValueError
        return self.embedding(embedding_idx)


def compute_forecast(predictor_model, initial_sequence, n_future_steps):
    """Input :  [Batch, Seqlen, Channels]
    Output : [Batch, Seqlen + n_future_steps, Channels]
    """
    sequence = initial_sequence
    for i in range(n_future_steps):
        model_out = predictor_model.forward(sequence)
        if isinstance(model_out, dict):
            model_out = model_out["regression_output"]
        last_timestep = model_out[:, [-1]]
        sequence = torch.cat([model_out, last_timestep], dim=1)
    return sequence


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
    activation: nn.Module = nn.ReLU(),
    dropout: float = 0.0,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_size, hidden_size, bias=False),
        nn.Dropout(dropout),
        Rearrange("batch seqlen channels -> batch channels seqlen"),
        nn.BatchNorm1d(hidden_size),
        Rearrange("batch channels seqlen -> batch seqlen channels"),
        activation,
    )


class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_size,
        tomean=None,
        tologvar=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

        if tomean is None:
            self.tomean = nn.Linear(latent_size, latent_size)
        else:
            self.tomean = tomean
        if tologvar is None:
            self.tologvar = nn.Linear(latent_size, latent_size)
        else:
            self.tologvar = tologvar

    def forward(self, *args, **kwargs):
        encoded = self.encoder(*args, **kwargs)
        latent_mean, latent_logstd = self.tomean(encoded), self.tologvar(encoded)
        latent = self.latent_sample(latent_mean, latent_logstd)
        reconstruction = self.decoder(latent)
        return {
            "reconstruction": reconstruction,
            "latent_mean": latent_mean,
            "latent_logvar": latent_logstd,
        }

    def latent_sample(self, mu, logvar):

        if self.training:
            # Convert the logvar to std
            std = (logvar * 0.5).exp()

            # the reparameterization trick
            return torch.distributions.Normal(loc=mu, scale=std).rsample()

            # Or if you prefer to do it without a torch.distribution...
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu


class FeedForwardVAE(VariationalAutoEncoder):
    def __init__(self, num_inputs, latent_size):
        encoder = nn.Sequential(
            NonLinear(num_inputs, num_inputs // 2, dropout=0.1),
        )
        tomean = nn.Linear(num_inputs // 2, latent_size)
        tologvar = nn.Linear(num_inputs // 2, latent_size)
        decoder = nn.Sequential(
            NonLinear(num_inputs // 2, num_inputs),
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_size=latent_size,
            tomean=tomean,
            tologvar=tologvar,
        )


class TcnVAE(VariationalAutoEncoder):
    def __init__(
        self,
        num_inputs,
        latent_size,
        sequence_length,
        num_channels=[16],
        dropout=0.0,
        kernel_size=2,
        stride=1,
        channels_last=False,
    ):
        TcnClass = TCNWrapper if channels_last else TemporalConvNet
        self.num_channels = num_channels
        self.latent_size = latent_size
        if stride > 1:
            flattened_size = num_channels[-1] * (
                sequence_length / (stride ** (len(num_channels)))
            )
        else:
            flattened_size = num_channels[-1] * sequence_length
        flattened_size = int(flattened_size)
        # if stride > 1:
        #     result = calc_tcn_outs(
        #         input_length=sequence_length,
        #         num_channels=num_channels,
        #         kernel_size=kernel_size,
        #         stride=stride,
        #     )
        #     flattened_size = int(num_channels[-1] * result[-1]["out_length"])
        encoder = nn.Sequential(
            TcnClass(
                num_inputs=num_inputs,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                stride=stride,
            ),
            Rearrange("batch seq chan -> batch (seq chan)"),
        )
        tomean = nn.Linear(flattened_size, latent_size)
        tologvar = nn.Linear(flattened_size, latent_size)

        decoder_num_channels = list(reversed(num_channels))
        decoder = nn.Sequential(
            nn.Linear(latent_size, flattened_size),
            Rearrange(
                "batch (seq chan) -> batch seq chan"
                if channels_last
                else "batch (chan seq) -> batch chan seq",
                seq=flattened_size // decoder_num_channels[0],
                chan=decoder_num_channels[0],
            ),
            TcnClass(
                num_inputs=decoder_num_channels[0],
                num_channels=decoder_num_channels,
                kernel_size=kernel_size,
                transposed=True,
                dropout=dropout,
                stride=1,
                upsample=stride,
            ),
            nn.Conv1d(decoder_num_channels[-1], num_inputs, 1),
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_size=latent_size,
            tomean=tomean,
            tologvar=tologvar,
        )


def vae_loss(recon_x, x, mu, logvar, variational_beta=1):
    recon_loss = F.binary_cross_entropy(
        recon_x.reshape(-1), x.reshape(-1), reduction="sum"
    )

    # You can look at the derivation of the KL term here https://arxiv.org/pdf/1907.08956.pdf
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return {
        "loss": recon_loss + variational_beta * kldivergence,
        "recon_loss": recon_loss,
        "kl_divergence": variational_beta * kldivergence,
    }


def VaeLoss(variational_beta=1):
    def f(*args, **kwargs):
        return vae_loss(*args, **kwargs, variational_beta=variational_beta)

    return f


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
        # Seqlen, channels
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ### Seqlen, channels -> Batch, Seqlen, channels -> Seqlen, Batch, channels
        # pe = pe.unsqueeze(0).transpose(0, 1)
        ### Seqlen, channels -> Batch, Seqlen, channels
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        #### ATTENTION: this assumes shape == (seq, Batch, channels)?
        # return x + self.pe[: x.size(0), :]
        # AND THIS SHOULD BE FOR (Battch, seq, channels) ?
        return x + self.pe[:, : x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        pe = torch.empty(max_len, d_model)
        ### Seqlen, channels -> Batch, Seqlen, channels -> Seqlen, Batch, channels
        # pe = pe.unsqueeze(0).transpose(0, 1)
        ### Seqlen, channels -> Batch, Seqlen, channels
        pe = pe.unsqueeze(0)
        nn.init.uniform_(pe, 0.02, 0.02)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        #### ATTENTION: this assumes shape == (seq, Batch, channels)?
        # return x + self.pe[: x.size(0), :]
        return x + self.pe[:, : x.size(1), :]


class CausalTransformer(nn.Module):
    def __init__(
        self,
        feature_size=256,
        feedforward_size=1024,
        num_layers=1,
        n_heads=4,
        dropout=0.1,
        positional_encoding="sine",
    ):
        super(CausalTransformer, self).__init__()
        self.feature_size = feature_size
        self.feedforward_size = feedforward_size

        self.src_mask = None
        pos_encoder_class = (
            SinePositionalEncoding
            if positional_encoding == "sine"
            else LearnedPositionalEncoding
        )
        self.pos_encoder = pos_encoder_class(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=n_heads,
            dropout=dropout,
            dim_feedforward=feedforward_size,
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


class TCNWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tcn = TemporalConvNet(*args, **kwargs)

    def forward(self, x):
        x = self.tcn(x.transpose(-1, -2))
        return x.transpose(-1, -2)


class TransformerForecaster(nn.Module):
    forecast = compute_forecast

    def __init__(
        self,
        in_size,
        transformer: CausalTransformer,
        embed_by_repetition=False,
        embedding=None,
    ):
        super().__init__()
        self.in_size = in_size
        self.embed_by_repetition = embed_by_repetition
        assert embed_by_repetition in [True, False]
        if self.embed_by_repetition:
            assert transformer.feature_size % in_size == 0
            self.repeat = transformer.feature_size // in_size
        elif embedding is None:
            self.embedding = NonLinear(in_size, transformer.feature_size)
        else:
            self.embedding = embedding
        self.transformer = transformer
        self.out_dim = transformer.feature_size

    def forward(self, x, return_dict=False):
        if self.embed_by_repetition:
            embedded = repeat(x, "B L C -> B L (repeat C)", repeat=self.repeat)
        else:
            embedded = self.embedding(x)
        out = self.transformer(embedded)
        if return_dict:
            return {"transformer_out": out, "transformer_in": embedded}
        else:
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
    forecast = compute_forecast

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


class FeatureScaler(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.factors = nn.Parameter(torch.ones(in_size), requires_grad=True)

    def forward(self, x):
        return x * self.factors


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
        hidden_sizes: list[int],
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
        self,
        autoencoder: FeedForwardAutoEncoder,
        n_classes: int,
        use_feature_scaler=False,
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
        if isinstance(features, dict):
            encoded = features["transformer_out"]
            out = {k: v.clone().detach() for k, v in features.items()}
        else:
            encoded = features
            out = dict()
        out.update({"regression_output": self.decoder(encoded)})
        return out

    # def forecast(self, *args, **kwargs):
    #     return compute_forecast(
    #         nn.Sequential(self.feature_scaler, self.feature_extractor, self.decoder),
    #         *args,
    #         **kwargs
    #     )
    def forecast(self, initial_sequence, n_future_steps):
        """Input :  [Batch, Seqlen, Channels]
        Output : [Batch, Seqlen + n_future_steps, Channels]
        """
        sequence = initial_sequence
        for i in range(n_future_steps):
            model_out = self.forward(sequence)
            if isinstance(model_out, dict):
                model_out = model_out["regression_output"]
            last_timestep = model_out[:, [-1]]
            sequence = torch.cat([model_out, last_timestep], dim=1)
        return sequence
