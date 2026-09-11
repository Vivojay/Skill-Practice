import torch
from deepseek_minimal.dsa import SparseMLA, select_tokens
from deepseek_minimal.models import v3, v32
from deepseek_minimal.models.prediction import MultiTokenModel


def test_selection_never_includes_future_and_all_keys_matches_mla():
    torch.manual_seed(7)
    layer = SparseMLA(top_k=99).double().eval()
    x = torch.randn(2,7,32,dtype=torch.float64)
    dense = layer._attention(x,None,0,True)[0]
    torch.testing.assert_close(layer(x),dense)
    scores = torch.randn(2,7,7)
    causal = torch.ones(7,7,dtype=torch.bool).tril().expand(2,-1,-1)
    selected = select_tokens(scores,causal,3)
    assert not (selected & ~causal).any()
    assert torch.equal(selected.sum(-1)[0],torch.tensor([1,2,3,3,3,3,3]))


def test_sparse_cache_and_separate_indexer_gradients():
    torch.manual_seed(3)
    layer = SparseMLA(top_k=3).double()
    x = torch.randn(2,7,32,dtype=torch.float64,requires_grad=True)
    y = layer(x)
    layer.index_objective.backward()
    assert x.grad is None and layer.q_down.weight.grad is None
    assert layer.indexer.q.weight.grad.abs().sum() > 0
    layer.zero_grad(set_to_none=True)
    y.square().sum().backward()
    assert layer.q_down.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in layer.indexer.parameters())
    layer.eval()
    ys, cache = [], None
    for t in range(7):
        out,cache,_ = layer.decode(x[:,t:t+1],cache)
        ys.append(out)
    torch.testing.assert_close(torch.cat(ys,1),layer(x))
    model = v32.build().double().eval()
    tokens = torch.randint(0,32,(1,7))
    a,cache = model.decode(tokens[:,:3])
    b,_ = model.decode(tokens[:,3:],cache)
    torch.testing.assert_close(torch.cat((a,b),1),model(tokens))


def test_integrated_mtp_sharing_and_shifted_causality():
    model = MultiTokenModel(v3.build()).double()
    tokens = torch.tensor([[1,2,3,4,5]])
    ntp,extra = model(tokens)
    changed = tokens.clone(); changed[:,3:] = 7
    ntp2,extra2 = model(changed)
    torch.testing.assert_close(ntp[:,:3],ntp2[:,:3])
    torch.testing.assert_close(extra[:,:2],extra2[:,:2])
    (ntp.square().mean()+extra.square().mean()).backward()
    assert model.base.embedding.weight.grad.abs().sum() > 0
    assert model.next_block.attention.q_down.weight.grad.abs().sum() > 0
    assert model.base.head.weight.grad.abs().sum() > 0
