import torch
from torch import nn
from typing import Optional


from .model_utils import *


class Regressor(nn.Module):
    def __init__(
        self,
        feature_extractor,
        n_outputs: int,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()
        if feature_dim is None:
            feature_dim = feature_extractor.out_dim

        self.feature_extractor = feature_extractor
        self.n_outputs = n_outputs
        self.feature_dim = feature_dim
        self.decoder = nn.Linear(feature_dim, n_outputs)

    def forward(self, x):
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
