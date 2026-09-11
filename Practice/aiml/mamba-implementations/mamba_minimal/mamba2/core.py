# SSD factorization adapted from state-spaces/mamba/modules/ssd_minimal.py.
# Copyright (c) 2024, Albert Gu and Tri Dao. Apache-2.0.
"""Discrete SSD: x already includes delta; a is log decay (delta * A).

x: (batch,T,H,P), a: (batch,T,H), B,C: (batch,T,H,N), state: (batch,H,P,N).
The block expands its single B/C group to H heads before calling these functions.
"""
import torch


def _state(x, B, state):
    if x.shape[1] == 0:
        raise ValueError("SSD requires at least one token")
    return x.new_zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[-1]) if state is None else state


def ssd_recurrent(x, a, B, C, state=None):
    h = _state(x, B, state)
    ys = []
    for t in range(x.shape[1]):
        h = a[:, t, :, None, None].exp() * h + x[:, t, :, :, None] * B[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", h, C[:, t]))
    return torch.stack(ys, 1), h


def decay_matrix(a):
    """(batch,T,H) -> (batch,H,T,T), L[t,s]=exp(sum(a[s+1:t+1])).

    Sum only each segment, avoiding subtraction of large cumulative sums.
    Mask BEFORE exponentiation, so an unused upper triangle cannot overflow.
    """
    length = a.shape[1]
    strict = torch.ones(length, length, dtype=torch.bool, device=a.device).tril(-1)
    terms = a.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, length)
    segments = terms.masked_fill(~strict, 0).cumsum(-2)
    return segments.masked_fill(~strict.fill_diagonal_(True), -torch.inf).exp()


def ssd_dense(x, a, B, C, state=None):
    """Tiny verification path: materializes the quadratic semiseparable matrix."""
    h = _state(x, B, state)
    L = decay_matrix(a)
    weights = torch.einsum("bthn,bshn->bhts", C, B) * L
    y = torch.einsum("bhts,bshp->bthp", weights, x)
    prefix = a.cumsum(1).exp()
    y = y + torch.einsum("bthn,bhpn->bthp", C, h) * prefix.unsqueeze(-1)
    final = prefix[:, -1, :, None, None] * h
    final = final + torch.einsum("bhs,bshp,bshn->bhpn", L[:, :, -1], x, B)
    return y, final


def ssd_chunked(x, a, B, C, state=None, chunk_size=16):
    """Within-chunk matrix products, with a sequential loop over chunk summaries.

    No padding: the final partial chunk has its actual length and decay.
    This Python implementation does not execute different chunks in parallel.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    h = _state(x, B, state)
    outputs = []
    for start in range(0, x.shape[1], chunk_size):
        xc, ac, bc, cc = (v[:, start:start+chunk_size] for v in (x, a, B, C))
        L = decay_matrix(ac)  # (batch,H,Q,Q), Q can be smaller than chunk_size
        weights = torch.einsum("bqhn,bshn->bhqs", cc, bc) * L
        within = torch.einsum("bhqs,bshp->bqhp", weights, xc)
        prefix = ac.cumsum(1).exp()
        incoming = torch.einsum("bqhn,bhpn->bqhp", cc, h) * prefix.unsqueeze(-1)
        outputs.append(within + incoming)
        summary = torch.einsum("bhs,bshp,bshn->bhpn", L[:, :, -1], xc, bc)
        h = prefix[:, -1, :, None, None] * h + summary
    return torch.cat(outputs, 1), h
