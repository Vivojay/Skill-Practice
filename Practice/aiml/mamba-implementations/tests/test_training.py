import io
import json

import pytest
import torch

from mamba_minimal.model import CopyModel
from mamba_minimal.task import selective_copy, recall_metrics


def test_task_has_no_recall_input_leakage():
    x, target = selective_copy(16, 32, torch.Generator().manual_seed(1001))
    for tokens, labels in zip(x, target):
        relevant = tokens[(tokens >= 1) & (tokens <= 4)]
        torch.testing.assert_close(relevant-1, labels[labels != -100])
        assert (tokens[labels != -100] == 0).all()
        assert (labels[:29] == -100).all()
        assert tokens[28] == 9
        assert (tokens[29:] == 0).all()
    # Only recall logits affect the objective.
    logits = torch.randn(16, 32, 4, requires_grad=True)
    recall_metrics(logits, target)[0].backward()
    assert (logits.grad[target == -100] == 0).all()
    assert logits.grad[target != -100].abs().sum() > 0


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["mamba1", "mamba2", "mamba3", "mamba1_constant"])
def test_fixed_batch_overfit(kind):
    torch.manual_seed(1)
    model = CopyModel(kind)
    data = selective_copy(4, 16, torch.Generator().manual_seed(1001), copies=2)
    assert data[1][data[1] != -100].unique().numel() >= 3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    initial, _ = recall_metrics(model(data[0]), data[1])
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = recall_metrics(model(data[0]), data[1])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final, accuracy = recall_metrics(model(data[0]), data[1])
    print(json.dumps(dict(model=kind, initial_loss=initial.item(), final_loss=final.item(),
                          fixed_batch_accuracy=accuracy.item(), updates=400, seed=1, data_seed=1001)))
    assert final < 0.05 and final < initial * 0.1
    assert accuracy == 1.0


@pytest.mark.parametrize("kind", ["mamba1", "mamba2", "mamba3", "mamba1_constant"])
def test_checkpoint_roundtrip(kind):
    torch.manual_seed(2)
    model = CopyModel(kind).eval()
    x, _ = selective_copy(2, 16, torch.Generator().manual_seed(4), copies=2)
    expected = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    reloaded = CopyModel(kind).eval()
    reloaded.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(reloaded(x), expected, atol=0, rtol=0)
