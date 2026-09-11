import pytest
import torch
from torch.autograd import gradcheck

from mamba_minimal.mamba3 import Mamba3, Mamba3State, siso_scan
from mamba_minimal.mamba2 import ssd_recurrent
from complex_oracle import complex_oracle


def inputs(length=4):
    torch.manual_seed(40)
    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64)
    x, dt, A = rand(1, length, 2, 2), torch.rand(1, length, 2, dtype=torch.float64) + 0.1, -rand(1, length, 2).abs()
    B, C = rand(1, length, 2, 4), rand(1, length, 2, 4)
    lam, theta = rand(1, length, 2).sigmoid(), rand(1, length, 2, 1)
    state = Mamba3State(rand(1, 2, 1), rand(1, 2, 2, 4), rand(1, 2, 4), rand(1, 2, 2))
    return (x, dt, A, B, C, lam, theta), state


@pytest.mark.parametrize("carried", [True, False])
def test_complex_oracle(carried):
    args, state = inputs()
    if not carried:
        state = Mamba3State(*(torch.zeros_like(v) for v in state))
    y, final = siso_scan(*args, state)
    expected, expected_h = complex_oracle(*args, state)
    torch.testing.assert_close(y, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final.ssm, expected_h, atol=1e-11, rtol=1e-10)


def test_numerical_gradients_including_history():
    args, state = inputs(2)
    tensors = tuple(v.requires_grad_() for v in (*args, *state))
    def flattened(*values):
        y, s = siso_scan(*values[:7], Mamba3State(*values[7:]))
        return (y, *s)
    assert gradcheck(flattened, tensors, eps=1e-6, atol=1e-5, rtol=1e-4)


def test_lambda_one_complex_euler_limit():
    args, state = inputs()
    x, dt, A, B, C, _, theta = args
    lam = torch.ones_like(dt)
    y, final = siso_scan(x, dt, A, B, C, lam, theta, state)
    expected, h = complex_oracle(x, dt, A, B, C, lam, theta, state)
    torch.testing.assert_close(y, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final.ssm, h, atol=1e-11, rtol=1e-10)
    # At lambda=1 arbitrary previous key/value must have no effect.
    changed = state._replace(key=100*state.key, value=-100*state.value)
    yy, _ = siso_scan(x, dt, A, B, C, lam, theta, changed)
    torch.testing.assert_close(y, yy, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("euler", [False, True])
def test_zero_rotation_limit(euler):
    args, state = inputs()
    x, dt, A, B, C, lam, theta = args
    # Zero phase is required to compare cached state coordinates directly.
    state = state._replace(phase=torch.zeros_like(state.phase))
    lam = torch.ones_like(lam) if euler else lam
    y, final = siso_scan(x, dt, A, B, C, lam, torch.zeros_like(theta), state)
    if euler:
        expected, h = ssd_recurrent(x*dt.unsqueeze(-1), dt*A, B, C, state.ssm)
    else:
        # Explicit real trapezoidal recurrence without phase or rotation helpers.
        h, previous = state.ssm, state.value.unsqueeze(-1)*state.key.unsqueeze(-2)
        outputs = []
        for t in range(x.shape[1]):
            current = x[:, t, :, :, None] * B[:, t, :, None, :]
            decay = (dt[:, t]*A[:, t]).exp()[..., None, None]
            h = decay * (h + ((1-lam[:, t])*dt[:, t])[..., None, None]*previous)
            h = h + (lam[:, t]*dt[:, t])[..., None, None]*current
            outputs.append((h*C[:, t, :, None, :]).sum(-1))
            previous = current
        expected = torch.stack(outputs, 1)
    torch.testing.assert_close(y, expected, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final.ssm, h, atol=1e-11, rtol=1e-10)


def test_block_projection_order_and_initialization():
    block = Mamba3(8, 4, head_dim=4).double()
    u = torch.randn(2, 8, dtype=torch.float64)
    z, x, dt, A, B, C, lam, theta = block._project(u)
    raw = block.in_proj(u)
    braw = raw[..., 2*block.inner:2*block.inner+4]
    expected = braw * (braw.square().mean(-1, keepdim=True)+1e-5).rsqrt()
    torch.testing.assert_close(B, expected.unsqueeze(-2).expand_as(B)+1)
    assert (A < 0).all() and (dt > 0).all() and (theta.abs() <= torch.pi).all()
    torch.testing.assert_close(block.C_bias, torch.ones_like(block.C_bias))
