import math
import torch
from deepseek_minimal.rotary import inverse_frequencies, rotate, attention_scale
from deepseek_minimal.mla import MLA
from deepseek_minimal.moe import MoE


def test_frequency_ramp_and_relative_positions():
    f = inverse_frequencies(16, factor=4, original_length=64, dtype=torch.float64)
    plain = inverse_frequencies(16, dtype=torch.float64)
    assert torch.all(f <= plain) and torch.all(f >= plain / 4)
    assert f[0] == plain[0] and f[-1] == plain[-1] / 4
    x, y = torch.randn(2, 1, 3, 16, dtype=torch.float64), torch.randn(2, 1, 3, 16, dtype=torch.float64)
    for interleaved in (False, True):
        def r(a, p):
            return rotate(a, torch.tensor([p]), interleaved=interleaved, factor=4, original_length=64)
        torch.testing.assert_close((r(x, 23)*r(y, 19)).sum(-1), (r(x, 4)*y).sum(-1))
    assert attention_scale(4) == (1+.1*math.log(4))**2
    assert attention_scale(4,.707) == (1+.0707*math.log(4))**2


def test_scaled_mla_cache_crosses_original_context():
    model = MLA(rotary=dict(factor=4, original_length=4)).double().eval()
    x = torch.randn(1, 9, 32, dtype=torch.float64)
    full = model(x)
    cache, pieces = None, []
    for t in range(9):
        y, cache, _ = model.decode(x[:, t:t+1], cache)
        pieces.append(y)
    torch.testing.assert_close(full, torch.cat(pieces, 1))


def test_group_selection_uses_bias_but_weights_do_not():
    model = MoE(width=4, routed=4, top_k=2, groups=2, top_groups=1,
                group_score='sum2', score='sigmoid', normalize=True, route_scale=2.)
    with torch.no_grad():
        model.router.weight.copy_(torch.eye(4))
        model.selection_bias.copy_(torch.tensor([0., 0., 2., 2.]))
    x = torch.tensor([[3., 2., 1., -1.]])
    scores, indices, weights = model.route(x)
    assert set(indices[0].tolist()) == {2, 3}
    expected = torch.tensor([1., -1.]).sigmoid()
    torch.testing.assert_close(weights[0], 2*expected/expected.sum())
    torch.testing.assert_close(scores, x.sigmoid())
