"""Direct complex physical-state oracle, independent of cumulative real rotation.

The official positive B/C rotation convention means physical transition
exp(dt*A - i*dt*theta). Readout is Re(conj(C_complex) * h_complex).
The sign of the learnable angular velocity is a coordinate convention.
"""
import torch
from torch.nn import functional as F


def complex_oracle(x, dt, A, B, C, lam, theta, state):
    def complexify(v):
        return torch.view_as_complex(v.reshape(*v.shape[:-1], -1, 2).contiguous())

    phase, h, key, value = state
    n_pairs = B.shape[-1] // 2
    full_phase = F.pad(phase, (0, n_pairs - phase.shape[-1]))
    # Convert the provided cumulative-rotary cache to physical coordinates.
    h = complexify(h) * torch.exp(-1j * full_phase).unsqueeze(-2)
    prev = complexify(key) * torch.exp(-1j * full_phase)
    outputs = []
    for t in range(x.shape[1]):
        w = F.pad(theta[:, t], (0, n_pairs - theta.shape[-1]))
        transition = torch.exp(dt[:, t, :, None] * (A[:, t, :, None] - 1j*w))
        h = transition.unsqueeze(-2) * (
            h + ((1-lam[:, t]) * dt[:, t])[..., None, None] * value.unsqueeze(-1) * prev.unsqueeze(-2))
        current = complexify(B[:, t])
        h = h + (lam[:, t]*dt[:, t])[..., None, None] * x[:, t, :, :, None] * current.unsqueeze(-2)
        outputs.append((h * complexify(C[:, t]).conj().unsqueeze(-2)).real.sum(-1))
        prev, value = current, x[:, t]
        full_phase = full_phase + dt[:, t, :, None] * w
    # Compare the final SSM state in the implementation's coordinates as well.
    rotated = h * torch.exp(1j * full_phase).unsqueeze(-2)
    return torch.stack(outputs, 1), torch.view_as_real(rotated).flatten(-2)
