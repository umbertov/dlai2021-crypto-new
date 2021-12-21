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
from typing import List, Dict, Tuple

from src.tcn import TemporalConvNet

from .model_utils import *


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
