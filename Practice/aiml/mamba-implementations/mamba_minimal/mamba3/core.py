# SISO ordering adapted from state-spaces/mamba tests/ops/triton/test_mamba3_siso.py.
# Copyright (c) 2025, Dao AI Lab, Goombalab. Apache-2.0.
"""Mamba-3 SISO educational subset, in the official positive-rotary gauge.

x: (batch,T,H,P), dt,A,lam: (batch,T,H); B,C: (batch,T,H,N).
theta: (batch,T,H,R), angular velocities AFTER pi*tanh; 2*R <= N.
B,C enter AFTER normalization and biases. No implicit activation inside the core.
"""
import math
from typing import NamedTuple

import torch


class Mamba3State(NamedTuple):
    phase: torch.Tensor  # (batch,H,R), cumulative angle, modulo 2*pi
    ssm: torch.Tensor  # (batch,H,P,N), state in rotating B/C coordinates
    key: torch.Tensor  # (batch,H,N), previous already-rotated B
    value: torch.Tensor  # (batch,H,P), previous x (not delta*x)


def rotate_pairs(v, phase):
    """Rotate the first R interleaved real pairs, leaving remaining entries alone."""
    r = phase.shape[-1]
    pairs = v[..., :2*r].unflatten(-1, (r, 2))
    real, imag = pairs.unbind(-1)
    c, s = phase.cos(), phase.sin()
    rotated = torch.stack((real*c - imag*s, real*s + imag*c), -1).flatten(-2)
    return torch.cat((rotated, v[..., 2*r:]), -1)


def siso_step(x, dt, A, B, C, lam, theta, state):
    """One token; same dimensions as scan with time removed. Functional state."""
    phase = torch.remainder(state.phase + dt.unsqueeze(-1) * theta, 2 * math.pi)
    key, query = rotate_pairs(B, phase), rotate_pairs(C, phase)
    alpha = (dt * A).exp()
    beta = (1 - lam) * dt * alpha
    gamma = lam * dt
    h = alpha[..., None, None] * state.ssm
    h = h + beta[..., None, None] * state.value.unsqueeze(-1) * state.key.unsqueeze(-2)
    h = h + gamma[..., None, None] * x.unsqueeze(-1) * key.unsqueeze(-2)
    y = torch.einsum("bhpn,bhn->bhp", h, query)
    return y, Mamba3State(phase, h, key, x)


def siso_scan(x, dt, A, B, C, lam, theta, state=None):
    if x.shape[1] == 0 or 2 * theta.shape[-1] > B.shape[-1]:
        raise ValueError("nonempty sequence and 2*rotation_pairs <= state_size required")
    if state is None:
        batch, _, heads, p = x.shape
        n = B.shape[-1]
        state = Mamba3State(x.new_zeros(batch, heads, theta.shape[-1]),
                            x.new_zeros(batch, heads, p, n),
                            x.new_zeros(batch, heads, n), x.new_zeros(batch, heads, p))
    ys = []
    for t in range(x.shape[1]):
        y, state = siso_step(*(v[:, t] for v in (x, dt, A, B, C, lam, theta)), state)
        ys.append(y)
    return torch.stack(ys, 1), state
