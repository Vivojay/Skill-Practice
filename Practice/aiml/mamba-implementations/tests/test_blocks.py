import pytest
import torch

from mamba_minimal.mamba1 import Mamba1
from mamba_minimal.mamba2 import Mamba2
from mamba_minimal.mamba3 import Mamba3


def test_mamba3_streaming_and_causality():
    check_block(Mamba3(8, 4, head_dim=4))


@pytest.mark.parametrize("kernel", [1, 4])
def test_mamba2_streaming_and_causality(kernel):
    check_block(Mamba2(8, 4, head_dim=4, kernel_size=kernel, chunk_size=3))


@pytest.mark.parametrize("selective", [True, False])
@pytest.mark.parametrize("kernel", [1, 4])
def test_mamba1_streaming_and_causality(selective, kernel):
    check_block(Mamba1(8, 4, kernel_size=kernel, selective=selective))


def check_block(block):
    torch.manual_seed(20)
    block = block.double().eval()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    for carried in (False, True):
        state = block.initial_state(x)
        if carried:
            state = type(state)(*(torch.randn_like(s) for s in state))
        original = tuple(s.clone() for s in state)
        y, final = block(x, state)
        ys, current = [], state
        for t in range(x.shape[1]):
            o, current = block.step(x[:, t], current)
            ys.append(o)
        torch.testing.assert_close(y, torch.stack(ys, 1), atol=1e-10, rtol=1e-9)
        first, middle = block(x[:, :3], state)
        second, last = block(x[:, 3:], middle)
        torch.testing.assert_close(y, torch.cat((first, second), 1), atol=1e-10, rtol=1e-9)
        for a, b, c in zip(final, current, last):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
            torch.testing.assert_close(a, c, atol=1e-10, rtol=1e-9)
        for a, b in zip(state, original):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        perturbed = x.clone()
        perturbed[:, 4:] += 10 * torch.randn_like(perturbed[:, 4:])
        yp, _ = block(perturbed, state)
        torch.testing.assert_close(y[:, :4], yp[:, :4], atol=1e-10, rtol=1e-9)
