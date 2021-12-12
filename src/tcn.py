# from https://github.com/locuslab/TCN/raw/master/TCN/tcn.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]  # .contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        super(TemporalBlock, self).__init__()

        self.chomp = Chomp1d(padding)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.net = nn.Sequential(
            self.conv1,
            self.chomp,
            self.activation,
            self.dropout,
            self.conv2,
            self.chomp,
            self.activation,
            self.dropout,
        )
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=2,
        dropout=0.2,
        stride=1,
        transposed=False,
        upsample=1,
    ):
        super().__init__()

        self.upsample = upsample
        assert not (upsample > 1 and stride > 1)
        self.stride = stride
        self.num_channels = num_channels
        self.num_inputs = num_inputs

        LayerClass = TemporalBlock if not transposed else TemporalBlock
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                LayerClass(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=dilation_size * (kernel_size - 1),
                    dropout=dropout,
                )
            ]
            if upsample > 1:
                layers.append(nn.Upsample(scale_factor=(upsample,)))
            if stride > 1:
                layers.append(nn.MaxPool1d(stride))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TransposeTemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super().__init__()

        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv1 = weight_norm(
            nn.ConvTranspose1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=0,
                dilation=dilation,
            )
        )

        self.conv2 = weight_norm(
            nn.ConvTranspose1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=1,
                padding=0,
                dilation=dilation,
            )
        )

        self.net = nn.Sequential(
            self.conv1,
            self.chomp,
            self.relu,
            self.dropout,
            self.conv2,
            self.chomp,
            self.relu,
            self.dropout,
        )
        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out
