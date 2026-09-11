from functools import partial

import pytest
import torch
from torch.autograd import gradcheck

from mamba_minimal.mamba2 import ssd_recurrent, ssd_dense, ssd_chunked


def inputs(length=5):
    torch.manual_seed(30)
    shapes = [(1, length, 2, 2), (1, length, 2), (1, length, 2, 3), (1, length, 2, 3), (1, 2, 2, 3)]
    values = [torch.randn(s, dtype=torch.float64) for s in shapes]
    values[1] = -values[1].abs()
    return tuple(v.requires_grad_() for v in values)


@pytest.mark.parametrize("length", [1, 5, 8])
@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 8, 16])
def test_paths_outputs_states_gradients(length, chunk):
    args = inputs(length)
    expected = ssd_recurrent(*args)
    # Include a loss on final state to catch lost/incorrect carried-state gradients.
    gy, gh = (torch.randn_like(t) for t in expected)
    ref_grad = torch.autograd.grad((expected[0] * gy).sum() + (expected[1] * gh).sum(), args)
    for fn in (ssd_dense, partial(ssd_chunked, chunk_size=chunk)):
        actual = fn(*args)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
        grads = torch.autograd.grad((actual[0] * gy).sum() + (actual[1] * gh).sum(), args)
        for a, b in zip(grads, ref_grad):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("fn", [ssd_recurrent, ssd_dense, partial(ssd_chunked, chunk_size=2)])
def test_numerical_gradients(fn):
    assert gradcheck(fn, inputs(3), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_dense_independent_scalar_products():
    x, a, B, C, initial = inputs(5)
    ys = torch.zeros_like(x)
    h = torch.zeros_like(initial)
    for b in range(x.shape[0]):
        for t in range(5):
            for head in range(2):
                for p in range(2):
                    for n in range(3):
                        v = initial[b, head, p, n] * a[b, :t+1, head].exp().prod()
                        for s in range(t+1):
                            v = v + B[b, s, head, n] * x[b, s, head, p] * a[b, s+1:t+1, head].exp().prod()
                        ys[b, t, head, p] += C[b, t, head, n] * v
                        if t == 4:
                            h[b, head, p, n] = v
    y, final = ssd_dense(x, a, B, C, initial)
    torch.testing.assert_close(y, ys, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(final, h, atol=1e-11, rtol=1e-10)


def test_large_negative_decay_no_upper_triangle_overflow():
    args = list(inputs(8))
    args[1] = torch.full_like(args[1], -1000)
    ref = ssd_recurrent(*args)
    for fn in (ssd_dense, partial(ssd_chunked, chunk_size=3)):
        for a, b in zip(fn(*args), ref):
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
