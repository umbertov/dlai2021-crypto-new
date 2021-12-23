import torch
from torch.nn import functional as F
import torch.nn as nn


class DynamicWeightCrossEntropy(nn.Module):
    def __init__(self, n_classes, decay=0.8):
        super().__init__()
        self.n_classes = n_classes
        weight = torch.ones(n_classes, dtype=torch.float, requires_grad=False)
        self.register_buffer("weight", weight)
        self.decay = decay

    def forward(self, logits, targets):
        value_counts = torch.ones_like(self.weight)
        for i in range(self.n_classes):
            value_counts[i] += (targets == i).sum()
        # obtain the inverse value counts
        new_weight = (value_counts.sum()) / value_counts
        # normalize so it sums to 1
        new_weight = new_weight / new_weight.sum()
        new_weight = torch.maximum(new_weight, torch.tensor(0.05))
        new_weight = new_weight / new_weight.sum()
        # update weights with smoothing
        new_weight = (self.decay) * self.weight + (1 - self.decay) * new_weight
        if not logits.requires_grad:
            return F.cross_entropy(logits, targets, weight=new_weight, reduction="mean")
        self.weight = new_weight
        return F.cross_entropy(logits, targets, weight=self.weight, reduction="mean")
