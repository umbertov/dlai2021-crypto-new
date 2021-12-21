import torch
from torch import nn

from .model_utils import *


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
        compression=1,
        channels_last=False,
        reconstruction_method="transpose_conv",
        residual=False,
    ):
        assert reconstruction_method in ("upsample", "transpose_conv")
        TcnClass = TCNWrapper if channels_last else TemporalConvNet
        self.num_channels = num_channels
        self.latent_size = latent_size
        flattened_size = int(
            num_channels[-1] * (sequence_length / (compression ** (len(num_channels))))
        )
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
                downsample=compression,
                residual=residual,
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
                transposed=reconstruction_method == "transpose_conv",
                stride=compression if reconstruction_method == "transpose_conv" else 1,
                dropout=dropout,
                upsample=compression if reconstruction_method == "upsample" else 1,
                residual=residual,
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
