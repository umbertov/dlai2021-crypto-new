import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.nn import functional as F
from functools import reduce
from operator import mul
from typing import Optional, Callable, Dict, Tuple, List

from src.tcn import TemporalConvNet


class LambdaLayer(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.forward = f


def NonLinear(
    in_size: int,
    hidden_size: int,
    activation: nn.Module = nn.ReLU(),
    dropout: float = 0.0,
    channels_last=True,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_size, hidden_size, bias=False),
        nn.Dropout(dropout),
        activation,
    )


def conv_output_shape(input_length, kernel_size=1, stride=1, padding=0, dilation=1):
    from math import floor

    h = floor(
        ((input_length + (padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )
    return h


class TCNWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.tcn = TemporalConvNet(*args, **kwargs)

    def forward(self, x):
        x = self.tcn(x.transpose(-1, -2))
        return x.transpose(-1, -2)


def calc_tcn_outs(input_length, num_channels, kernel_size=2, stride=1):
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
        dilation_size = 2 ** i
        inum_channels = 0 if i == 0 else num_channels[i - 1]
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
                "inum_channels": inum_channels,
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
    for arg, num_classes in zip(tup, names_to_classes.values()):
        result += arg * prev_classes
        prev_classes = num_classes
    return result


def dict_to_index(dic, names_to_classes):
    result = 0
    prev_classes = 1
    for name, num_classes in names_to_classes.items():
        arg = dic[name]
        result += arg * prev_classes
        prev_classes = num_classes
    return result


class CartesianProdEmbedding(nn.Module):
    def __init__(self, hidden_size, **names_to_num_classes):
        super().__init__()
        self.names_to_classes = names_to_num_classes
        self.n_variables = len(names_to_num_classes)
        self.total_n_embeddings = reduce(mul, names_to_num_classes.items())
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


class DictEmbedder(nn.Module):
    def __init__(self, key2hiddensize: Dict[str, Tuple[int, int]]):
        self.k2h = key2hiddensize
        self.modules_dict = dict()
        for key, (in_size, hidden_size) in key2hiddensize.items():
            self.modules_dict[key] = NonLinear(in_size=in_size, hidden_size=hidden_size)
        self.modules = nn.ModuleList(list(self.modules_dict.items()))

    def forward(self, **keys2tensors) -> Dict[str, torch.Tensor]:
        if not all(k in self.modules_dict.keys() for k in keys2tensors):
            raise KeyError
        res = {
            key: self.modules_dict[key](input_tensor)
            for key, input_tensor in keys2tensors.items()
        }
        return res


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
