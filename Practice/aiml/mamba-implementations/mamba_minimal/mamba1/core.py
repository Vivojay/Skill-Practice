"""Algorithm 2, with the released implementation's exponential-Euler input term."""
import torch


def selective_scan(x, delta, A, B, C, state=None):
    """x, delta: (batch,time,inner); A: (inner,N); B,C: (batch,time,N).

    State: (batch,inner,N). Returns (output, final_state), both differentiable.
    delta must be positive and A negative. D/gating belong to the block.
    """
    if x.shape[1] == 0:
        raise ValueError("scan requires at least one token")
    h = x.new_zeros(x.shape[0], x.shape[2], A.shape[-1]) if state is None else state
    outputs = []
    for t in range(x.shape[1]):
        dt = delta[:, t, :, None]  # (batch,inner,1)
        h = torch.exp(dt * A) * h + dt * B[:, t, None, :] * x[:, t, :, None]
        outputs.append((h * C[:, t, None, :]).sum(-1))
    return torch.stack(outputs, dim=1), h
