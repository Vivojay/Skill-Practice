import pytest
import torch
from torch.nn import functional as F
from deepseek_minimal.models import v1, v2, v3


@pytest.mark.parametrize('build', [v1.build, lambda: v1.build(kv_heads=2),
                                   lambda: v1.build(moe=True), v2.build, v3.build])
def test_full_decoder_cache_causality_generation_and_training(build):
    torch.manual_seed(12)
    model = build().double().eval()
    tokens = torch.randint(0, 32, (2,7))
    full = model(tokens, offset=9)
    first, caches = model.decode(tokens[:, :3], start=9)
    last, caches = model.decode(tokens[:, 3:], caches)
    torch.testing.assert_close(torch.cat((first,last),1), full, atol=1e-10, rtol=1e-10)
    changed = tokens.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4])
    generated = tokens.clone()
    for _ in range(3):
        generated = torch.cat((generated,model(generated)[:, -1].argmax(-1,keepdim=True)),1)
    assert torch.equal(generated, model.generate(tokens,3))
    assert sum(c.nbytes for c in caches) > 0
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.015)
    initial = F.cross_entropy(model(tokens).flatten(0,1),tokens.flatten()).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(tokens).flatten(0,1),tokens.flatten())
        loss.backward()
        optimizer.step()
        model.finish_step()
    assert loss.item() < initial * .1


def test_v3_decode_never_updates_routing_and_rejects_bad_caches():
    model = v3.build().eval()
    before = {n:b.clone() for n,b in model.named_buffers()}
    _, cache = model.decode(torch.tensor([[1,2,3]]))
    model.generate(torch.tensor([[1,2,3]]),3)
    for n,b in model.named_buffers():
        assert torch.equal(before[n],b)
    with pytest.raises(ValueError):
        model.decode(torch.tensor([[4]]),cache,start=1)
    with pytest.raises(ValueError):
        model.decode(torch.tensor([[4]]),cache[:1])
