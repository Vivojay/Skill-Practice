# Architecture/initialization adapted from state-spaces/mamba/mamba2_simple.py.
# Copyright (c) 2024, Tri Dao, Albert Gu. Apache-2.0.
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm
from .core import ssd_chunked, ssd_dense, ssd_recurrent


class Mamba2State(NamedTuple):
    conv: torch.Tensor  # (batch,inner+2*N,kernel-1), projected x/B/C history
    ssm: torch.Tensor  # (batch,H,P,N)


class Mamba2(nn.Module):
    """One shared B/C group, multiple value heads, gate then RMSNorm."""

    def __init__(self, width, state_size=8, expand=2, head_dim=8, kernel_size=4, chunk_size=16):
        super().__init__()
        self.inner, self.state_size = width * expand, state_size
        if min(width, state_size, expand, head_dim, kernel_size, chunk_size) < 1 or self.inner % head_dim:
            raise ValueError("positive dimensions and inner divisible by head_dim required")
        self.head_dim, self.heads = head_dim, self.inner // head_dim
        self.kernel_size, self.chunk_size = kernel_size, chunk_size
        conv_dim = self.inner + 2 * state_size
        self.in_proj = nn.Linear(width, 2 * self.inner + 2 * state_size + self.heads, bias=False)
        self.conv = nn.Conv1d(conv_dim, conv_dim, kernel_size, groups=conv_dim)
        dt = torch.exp(torch.rand(self.heads) * math.log(100) + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = RMSNorm(self.inner)
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def initial_state(self, x):
        return Mamba2State(x.new_zeros(x.shape[0], self.inner + 2 * self.state_size, self.kernel_size - 1),
                           x.new_zeros(x.shape[0], self.heads, self.head_dim, self.state_size))

    def _split_xbc(self, xbc):
        x, B, C = xbc.split([self.inner, self.state_size, self.state_size], -1)
        x = x.unflatten(-1, (self.heads, self.head_dim))
        B = B.unsqueeze(-2).expand(*B.shape[:-1], self.heads, self.state_size)
        C = C.unsqueeze(-2).expand_as(B)
        return x, B, C

    def _output(self, y, x, z):
        y = (y + self.D[:, None] * x).flatten(-2)
        return self.out_proj(self.norm(y * F.silu(z)))

    def forward(self, u, state=None, path="chunked"):
        if u.shape[1] == 0:
            raise ValueError("block requires at least one token")
        state = self.initial_state(u) if state is None else state
        z, xbc, dt = self.in_proj(u).split([self.inner, self.inner + 2 * self.state_size, self.heads], -1)
        history = torch.cat((state.conv, xbc.transpose(1, 2)), -1)
        x, B, C = self._split_xbc(F.silu(self.conv(history).transpose(1, 2)))
        dt = F.softplus(dt + self.dt_bias)
        args = (x * dt.unsqueeze(-1), dt * -self.A_log.exp(), B, C, state.ssm)
        if path == "chunked":
            y, h = ssd_chunked(*args, chunk_size=self.chunk_size)
        elif path == "recurrent":
            y, h = ssd_recurrent(*args)
        elif path == "dense":
            y, h = ssd_dense(*args)
        else:
            raise ValueError("path must be recurrent, dense, or chunked")
        conv_state = history[..., -(self.kernel_size - 1):] if self.kernel_size > 1 else history[..., :0]
        return self._output(y, x, z), Mamba2State(conv_state, h)

    def step(self, u, state=None):
        state = self.initial_state(u) if state is None else state
        z, xbc, dt = self.in_proj(u).split([self.inner, self.inner + 2 * self.state_size, self.heads], -1)
        window = torch.cat((state.conv, xbc.unsqueeze(-1)), -1)
        x, B, C = self._split_xbc(F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias))
        dt = F.softplus(dt + self.dt_bias)
        h = (dt * -self.A_log.exp()).exp()[..., None, None] * state.ssm
        h = h + (x * dt.unsqueeze(-1)).unsqueeze(-1) * B.unsqueeze(-2)
        y = torch.einsum("bhpn,bhn->bhp", h, C)
        return self._output(y, x, z), Mamba2State(window[..., 1:], h)
