# Architecture and initialization adapted from state-spaces/mamba.
# Copyright (c) 2023, Tri Dao, Albert Gu. Apache-2.0; see THIRD_PARTY_NOTICES.md.
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .core import selective_scan


class Mamba1State(NamedTuple):
    conv: torch.Tensor  # (batch,inner,kernel-1), oldest first, pre-convolution
    ssm: torch.Tensor  # (batch,inner,N)


class Mamba1(nn.Module):
    def __init__(self, width, state_size=8, expand=2, kernel_size=4, selective=True):
        super().__init__()
        if min(width, state_size, expand, kernel_size) < 1:
            raise ValueError("dimensions must be positive")
        self.inner, self.state_size = width * expand, state_size
        self.kernel_size, self.selective = kernel_size, selective
        self.rank = math.ceil(width / 16)
        self.in_proj = nn.Linear(width, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, kernel_size, groups=self.inner)
        self.dt_proj = nn.Linear(self.rank, self.inner)
        if selective:
            self.x_proj = nn.Linear(self.inner, self.rank + 2 * state_size, bias=False)
        else:
            # Learned constants remove selection only inside the SSM.
            self.constant = nn.Parameter(torch.randn(self.rank + 2 * state_size) / math.sqrt(self.inner))
        self.A_log = nn.Parameter(torch.arange(1, state_size + 1).float().log().repeat(self.inner, 1))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, width, bias=False)
        nn.init.uniform_(self.dt_proj.weight, -self.rank**-0.5, self.rank**-0.5)
        dt = torch.exp(torch.rand(self.inner) * math.log(0.1 / 0.001) + math.log(0.001)).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def initial_state(self, x):
        return Mamba1State(x.new_zeros(x.shape[0], self.inner, self.kernel_size - 1),
                           x.new_zeros(x.shape[0], self.inner, self.state_size))

    def _parameters_from(self, x):
        p = self.x_proj(x) if self.selective else self.constant.expand(*x.shape[:-1], -1)
        dt, B, C = p.split([self.rank, self.state_size, self.state_size], -1)
        return F.softplus(self.dt_proj(dt)), B, C

    def forward(self, u, state=None):
        """u: (batch,time,width). Return (y, state); no in-place cache mutation."""
        if u.shape[1] == 0:
            raise ValueError("block requires at least one token")
        state = self.initial_state(u) if state is None else state
        x, z = self.in_proj(u).chunk(2, -1)
        history = torch.cat((state.conv, x.transpose(1, 2)), -1)
        x = F.silu(self.conv(history).transpose(1, 2))
        dt, B, C = self._parameters_from(x)
        y, h = selective_scan(x, dt, -self.A_log.exp(), B, C, state.ssm)
        y = self.out_proj((y + self.D * x) * F.silu(z))
        return y, Mamba1State(history[..., -(self.kernel_size - 1):] if self.kernel_size > 1 else history[..., :0], h)

    def step(self, u, state=None):
        """u: (batch,width); state includes the causal convolution history."""
        state = self.initial_state(u) if state is None else state
        x, z = self.in_proj(u).chunk(2, -1)
        window = torch.cat((state.conv, x.unsqueeze(-1)), -1)
        x = F.silu((window * self.conv.weight[:, 0]).sum(-1) + self.conv.bias)
        dt, B, C = self._parameters_from(x)
        h = (dt.unsqueeze(-1) * -self.A_log.exp()).exp() * state.ssm
        h = h + (dt * x).unsqueeze(-1) * B.unsqueeze(1)
        y = (h * C.unsqueeze(1)).sum(-1) + self.D * x
        return self.out_proj(y * F.silu(z)), Mamba1State(window[..., 1:], h)
