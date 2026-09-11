# Architecture/initialization adapted from state-spaces/mamba/modules/mamba3.py.
# Copyright (c) 2026, Dao AI Lab, Goombalab. Apache-2.0.
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..norm import RMSNorm
from .core import Mamba3State, siso_scan, siso_step


class Mamba3(nn.Module):
    """SISO only, one B/C group, half the state coordinates rotated.

    Follows reference default is_outproj_norm=False: no convolution, no x SiLU,
    no post-gate normalization. The shared wrapper supplies residual pre-norm.
    """

    def __init__(self, width, state_size=8, expand=2, head_dim=8):
        super().__init__()
        self.inner, self.state_size = width * expand, state_size
        if min(width, expand, head_dim) < 1 or self.inner % head_dim or state_size < 4 or state_size % 4:
            raise ValueError("inner must divide into heads; state_size must be a positive multiple of 4")
        self.head_dim, self.heads = head_dim, self.inner // head_dim
        self.pairs = state_size // 4
        self.in_proj = nn.Linear(width, 2*self.inner + 2*state_size + 3*self.heads + self.pairs, bias=False)
        dt = torch.exp(torch.rand(self.heads) * math.log(100) + math.log(0.001)).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.heads, state_size))
        self.C_bias = nn.Parameter(torch.ones(self.heads, state_size))
        self.B_norm, self.C_norm = RMSNorm(state_size), RMSNorm(state_size)
        self.D = nn.Parameter(torch.ones(self.heads))
        self.out_proj = nn.Linear(self.inner, width, bias=False)

    def initial_state(self, x):
        b = x.shape[0]
        return Mamba3State(x.new_zeros(b, self.heads, self.pairs),
                            x.new_zeros(b, self.heads, self.head_dim, self.state_size),
                            x.new_zeros(b, self.heads, self.state_size),
                            x.new_zeros(b, self.heads, self.head_dim))

    def _project(self, u):
        z, x, B, C, dt, a, trap, angles = self.in_proj(u).split(
            [self.inner, self.inner, self.state_size, self.state_size,
             self.heads, self.heads, self.heads, self.pairs], -1)
        x = x.unflatten(-1, (self.heads, self.head_dim))
        # heavy_tail(a) = 1+a for a>=0, 1/(1-a) otherwise; reference floor 1e-4.
        A = -(a.clamp_min(0) + (1 - a.clamp_max(0)).reciprocal()).clamp_min(1e-4)
        dt = F.softplus(dt + self.dt_bias)
        B = self.B_norm(B).unsqueeze(-2) + self.B_bias  # (...,H,N): normalize BEFORE bias
        C = self.C_norm(C).unsqueeze(-2) + self.C_bias
        theta = (math.pi * angles.tanh()).unsqueeze(-2).expand(*angles.shape[:-1], self.heads, self.pairs)
        return z, x, dt, A, B, C, trap.sigmoid(), theta

    def _output(self, y, x, z):
        return self.out_proj((y + self.D[:, None] * x).flatten(-2) * F.silu(z))

    def forward(self, u, state=None):
        z, x, dt, A, B, C, lam, theta = self._project(u)
        y, state = siso_scan(x, dt, A, B, C, lam, theta, state)
        return self._output(y, x, z), state

    def step(self, u, state=None):
        state = self.initial_state(u) if state is None else state
        z, x, dt, A, B, C, lam, theta = self._project(u)
        y, state = siso_step(x, dt, A, B, C, lam, theta, state)
        return self._output(y, x, z), state
