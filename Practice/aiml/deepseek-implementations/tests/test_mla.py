import math
import pytest
import torch
from deepseek_minimal.mla import MLA, rope


@pytest.mark.parametrize('offset', [0, 13])
def test_expanded_compressed_prefill_and_continuation(offset):
    torch.manual_seed(3)
    model = MLA(width=8, heads=2, content=3, positional=2, value=4, kv_rank=3, q_rank=4).double().eval()
    x = torch.randn(2,7,8,dtype=torch.float64)
    ref, expanded_cache, ref_scores = model.decode(x,start=offset,expanded=True)
    actual, cache, scores = model.decode(x,start=offset)
    torch.testing.assert_close(scores, ref_scores, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(actual, ref, atol=2e-12, rtol=2e-12)
    for chunks in [[1]*7, [3,1,3]]:
        prefix, ys, pos = None, [], offset
        for size in chunks:
            y,prefix,_ = model.decode(x[:,pos-offset:pos-offset+size],prefix,start=pos)
            ys.append(y); pos += size
        torch.testing.assert_close(torch.cat(ys,1),ref,atol=2e-12,rtol=2e-12)
    assert cache.nbytes == 2*7*(3+2)*8
    assert expanded_cache.nbytes == 2*7*2*(3+2+4)*8
    assert not cache.latent.requires_grad
    with pytest.raises(ValueError): model.decode(x[:,:1],cache,start=offset)
    with pytest.raises(ValueError): model.decode(x[:,:1],cache,expanded=True)


def test_rope_manual_rotation_and_causal_oracle():
    x = torch.tensor([[[[2.,3.]]]],dtype=torch.float64)
    rotated = rope(x,torch.tensor([1]))
    torch.testing.assert_close(rotated.flatten(), torch.tensor([2*math.cos(1)-3*math.sin(1),2*math.sin(1)+3*math.cos(1)],dtype=torch.float64))
    torch.manual_seed(4)
    model = MLA(width=4,heads=2,content=2,positional=2,value=2,kv_rank=2,q_rank=2).double()
    x = torch.randn(1,4,4,dtype=torch.float64,requires_grad=True)
    y = model(x)
    grad = torch.autograd.grad(y[:,1].sum(),x)[0]
    assert grad[:,2:].eq(0).all()
    # Independent scalar head/key loops, no attention einsums.
    qc, qr, c, kr = model._project(x,0)
    k, v = model.k_up(c).view(4,2,2), model.v_up(c).view(4,2,2)
    rows=[]
    for t in range(4):
        heads=[]
        for h in range(2):
            logits=torch.stack([(qc[0,t,h].dot(k[s,h])+qr[0,t,h].dot(kr[0,s]))/2 for s in range(t+1)])
            heads.append(sum(p*v[s,h] for s,p in enumerate(logits.softmax(0))))
        rows.append(model.out(torch.cat(heads)))
    torch.testing.assert_close(y[0],torch.stack(rows),atol=2e-12,rtol=2e-12)


def test_attention_gradcheck():
    torch.manual_seed(9)
    model=MLA(width=4,heads=1,content=2,positional=2,value=2,kv_rank=2,q_rank=2).double()
    x=torch.randn(1,3,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(model,(x,),atol=1e-5)
