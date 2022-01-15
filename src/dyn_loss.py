import torch
from torch.nn import functional as F
import torch.nn as nn


class DynamicWeightCrossEntropy(nn.Module):
    def __init__(self, n_classes, decay=0.8, minimum_weight=0.1, label_smoothing=0.0):
        super().__init__()
        self.n_classes = n_classes
        weight = torch.ones(n_classes, dtype=torch.float, requires_grad=False)
        self.register_buffer("weight", weight)
        self.decay = decay
        self.minimum_weight = minimum_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        value_counts = torch.ones_like(self.weight)
        # is there a better way to do this? it doesn't really hurt performance though
        for i in range(self.n_classes):
            value_counts[i] += (targets == i).sum()
        # obtain the inverse value counts
        new_weight = (value_counts.sum()) / value_counts
        # normalize so it sums to 1
        new_weight = new_weight / new_weight.sum()
        # cap the minimum weight to prevent weights that are too small (leads to nan/infinity)
        new_weight = torch.maximum(new_weight, torch.tensor(self.minimum_weight))
        # re-normalize
        new_weight = new_weight / new_weight.sum()
        # update weights with smoothing
        new_weight = (self.decay) * self.weight + (1 - self.decay) * new_weight
        # trick to only update weight estimation during training
        if logits.requires_grad:
            self.weight = new_weight
        return F.cross_entropy(
            logits,
            targets,
            weight=new_weight,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )
