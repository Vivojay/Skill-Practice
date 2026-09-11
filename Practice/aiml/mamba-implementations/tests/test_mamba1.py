import torch
from torch.autograd import gradcheck

from mamba_minimal.mamba1 import Mamba1, selective_scan


def test_core_unrolled_reference_and_gradients():
    torch.manual_seed(10)
    x = torch.randn(1, 4, 2, dtype=torch.float64, requires_grad=True)
    dt = torch.rand_like(x, requires_grad=True)
    A = (-torch.rand(2, 3, dtype=torch.float64)).requires_grad_()
    B = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    C = torch.randn_like(B, requires_grad=True)
    initial = torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True)
    y, h = selective_scan(x, dt, A, B, C, initial)
    refs = []
    for t in range(4):
        # Closed-form expansion, independently summing each past contribution.
        s = (dt[:, :t+1].sum(1).unsqueeze(-1) * A).exp() * initial
        for j in range(t + 1):
            decay = (dt[:, j+1:t+1].sum(1).unsqueeze(-1) * A).exp()
            s = s + decay * (dt[:, j] * x[:, j]).unsqueeze(-1) * B[:, j, None]
        refs.append((s * C[:, t, None]).sum(-1))
    torch.testing.assert_close(y, torch.stack(refs, 1), atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(h, s, atol=1e-11, rtol=1e-10)
    assert gradcheck(selective_scan, (x, dt, A, B, C, initial), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_initialization():
    m = Mamba1(16)
    torch.testing.assert_close(-m.A_log.exp(), -torch.arange(1, 9).float().repeat(32, 1))
    dt = torch.nn.functional.softplus(m.dt_proj.bias)
    assert (dt >= 0.001).all() and (dt <= 0.1).all()
    assert m.dt_proj.weight.abs().max() <= m.rank**-0.5
