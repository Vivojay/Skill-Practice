import copy
import math
import torch
from deepseek_minimal.mtp import MTPModel, prediction_losses, supervision


def test_offsets_masks_and_reduction():
    tokens=torch.tensor([[1,2,3,4,5],[2,3,4,0,0]])
    lengths=torch.tensor([5,3])
    targets,mask=supervision(tokens,lengths,1)
    assert targets.tolist()==[[3,4,5],[4,0,0]]
    assert mask.tolist()==[[True,True,True],[True,False,False]]
    ntp=torch.zeros(2,5,8,dtype=torch.float64,requires_grad=True)
    mtp=torch.zeros(2,4,8,dtype=torch.float64,requires_grad=True)
    a,b=prediction_losses(ntp,mtp,tokens,lengths)
    assert abs(a.item()-math.log(8))<1e-12
    assert abs(b.item()-4/6*math.log(8))<1e-12
    (a+b).backward()
    assert mtp.grad[1,1:].eq(0).all()


def test_sequential_causality_and_shared_gradients():
    torch.manual_seed(5)
    model=MTPModel(vocab=12,width=8,heads=2).double()
    tokens=torch.tensor([[1,2,3,4,5,6]])
    ntp,mtp=model(tokens)
    changed=tokens.clone(); changed[:,4:]=torch.tensor([8,9])
    n2,m2=model(changed)
    torch.testing.assert_close(ntp[:,:4],n2[:,:4],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(mtp[:,:3],m2[:,:3],atol=1e-12,rtol=1e-12)
    # The next input token really participates in the sequential module.
    changed=tokens.clone(); changed[:,1]=9
    assert not torch.allclose(model(changed)[1][:,0],mtp[:,0])
    mtp.square().sum().backward()
    assert model.base.embedding.weight.grad.abs().sum()>0
    assert model.base.head.weight.grad.abs().sum()>0
    assert model.base.block.attention.in_proj_weight.grad.abs().sum()>0


def test_equal_starting_base_and_tiny_overfit():
    torch.manual_seed(10)
    model=MTPModel(vocab=8,width=16,heads=2)
    other=copy.deepcopy(model)
    tokens=torch.tensor([[1,2,3,4,5,6],[3,4,5,6,7,0]])
    lengths=torch.tensor([6,6])
    torch.testing.assert_close(model(tokens,False)[0],other(tokens)[0])
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    for _ in range(65):
        optimizer.zero_grad()
        a,b=prediction_losses(*model(tokens),tokens,lengths)
        (a+.3*b).backward(); optimizer.step()
    assert a.item()<.02 and b.item()<.03
