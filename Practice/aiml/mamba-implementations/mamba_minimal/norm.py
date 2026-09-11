import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize the last dimension; preserve float64 for mathematical checks."""

    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight
