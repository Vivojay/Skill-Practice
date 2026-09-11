import pytest
import torch
from deepseek_minimal.compressed_attention import Compressor, CompressedAttention
from deepseek_minimal.hyperconnections import HyperConnection, sinkhorn
from deepseek_minimal.models import v4


def test_sinkhorn_constraints_and_residual_scalar_oracle():
    torch.manual_seed(5)
    scores = torch.randn(2,3,3,dtype=torch.float64,requires_grad=True)
    matrix = sinkhorn(scores,40)
    torch.testing.assert_close(matrix.sum(-1),torch.ones(2,3,dtype=torch.float64))
    torch.testing.assert_close(matrix.sum(-2),torch.ones(2,3,dtype=torch.float64))
    assert matrix.min() > 0
    assert torch.autograd.gradcheck(lambda s:sinkhorn(s,10),(scores,))
    layer = HyperConnection(4,3).double()
    x = torch.randn(1,2,3,4,dtype=torch.float64)
    h,post,res = layer.split(x)
    actual = layer.combine(x,h.square(),post,res)
    expected = torch.stack([sum(res[...,i,j,None]*x[...,j,:] for j in range(3)) + post[...,i,None]*h.square() for i in range(3)],-2)
    torch.testing.assert_close(actual,expected)


def test_channelwise_compression_uses_previous_and_current_blocks():
    module = Compressor(4,3,2,True).double()
    x = torch.randn(1,4,4,dtype=torch.float64)
    values = torch.cat((module.previous(x[:,:2]),module.current(x[:,2:])),1)
    scores = torch.cat((module.previous_gate(x[:,:2])+module.previous_bias,module.gate(x[:,2:])+module.bias),1)
    rows = []
    for channel in range(3):
        p = scores[0,:,channel].softmax(0)
        rows.append(sum(p[j]*values[0,j,channel] for j in range(4)))
    expected = module.norm(torch.stack(rows).view(1,1,3))
    torch.testing.assert_close(module(x),expected)


@pytest.mark.parametrize('ratio,overlap',[(0,False),(2,True),(8,False)])
def test_compressed_attention_causal_cache_at_block_boundaries(ratio,overlap):
    torch.manual_seed(1)
    layer = CompressedAttention(ratio=ratio,overlap=overlap).double().eval()
    x = torch.randn(2,13,32,dtype=torch.float64,requires_grad=True)
    full = layer(x,offset=7)
    grad = torch.autograd.grad(full[:,5].sum(),x)[0]
    assert grad[:,6:].eq(0).all()
    cache,parts = None,[]
    start = 7
    for left,right in ((0,1),(1,7),(7,8),(8,13)):
        y,cache,_ = layer.decode(x[:,left:right],cache,start)
        parts.append(y); start += right-left
    torch.testing.assert_close(torch.cat(parts,1),full)
    assert cache.window.shape[1] <= 4
    assert cache.entries.shape[1] == (13//ratio if ratio else 0)
    assert cache.tail.shape[1] < 2*ratio if ratio else cache.tail.shape[1] == 0
    assert cache.nbytes > 0


def test_v4_model_mtp_hash_routing_and_cache():
    torch.manual_seed(4)
    model = v4.build().double().eval()
    tokens = torch.randint(0,32,(1,9))
    full = model(tokens)
    prefix,caches = model.decode(tokens[:,:4])
    rest,caches = model.decode(tokens[:,4:],caches)
    torch.testing.assert_close(torch.cat((prefix,rest),1),full)
    expected = model.blocks[0].hash_table[tokens[:,4:]].reshape(-1,2)
    assert torch.equal(model.blocks[0].routing['indices'],expected)
    # Construct before running the model, so copied modules hold no live graph.
    mtp = v4.MultiTokenModel(v4.build()).double()
    ntp,extra = mtp(tokens)
    (ntp.square().mean()+extra.square().mean()).backward()
    assert mtp.base.blocks[0].attention.compressor.current.weight.grad.abs().sum() > 0


def test_window_attention_sink_is_zero_valued_and_output_is_derotated():
    layer = CompressedAttention(width=4,heads=1,dim=4,positional=4,ratio=0,
                                groups=1,out_rank=4).double()
    with torch.no_grad():
        layer.q_up.weight.zero_()
        layer.kv.weight.copy_(torch.eye(4))
        layer.output_groups[0].weight.copy_(torch.eye(4))
        layer.out.weight.copy_(torch.eye(4))
    x = torch.randn(1,1,4,dtype=torch.float64)
    # One real key and a sink with the same zero logit: exactly half the value.
    # The inverse rotation must cancel the key's absolute position here.
    torch.testing.assert_close(layer(x,offset=19),layer.kv_norm(x)/2)
