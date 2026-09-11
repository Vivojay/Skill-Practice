import math
import torch
from deepseek_minimal.grpo import (group_advantages, grpo_loss, response_mask,
    frozen_copy, sample_completions, completion_logp)
from deepseek_minimal.mtp import TinyLM


def test_objective_scalar_oracle_and_detachment():
    ratios=[1.5,.5,1.1,.9]
    logp=torch.tensor(ratios,dtype=torch.float64).log().reshape(1,2,2).requires_grad_()
    old=torch.zeros_like(logp,requires_grad=True)
    ref=torch.full_like(logp,-.2,requires_grad=True)
    advantages=torch.tensor([[1.,-1.]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[[True,True],[True,False]]])
    loss,info=grpo_loss(logp,old,ref,advantages,mask,beta=.1)
    sums=[]
    for i in range(2):
        terms=[]
        for t in range(2):
            if mask[0,i,t]:
                r=ratios[i*2+t]; a=[1.,-1.][i]
                d=-.2-math.log(r)
                terms.append(min(r*a,min(max(r,.8),1.2)*a)-.1*(math.exp(d)-d-1))
        sums.append(sum(terms)/len(terms))
    assert abs(loss.item()+sum(sums)/2)<1e-12
    loss.backward()
    assert old.grad is None and ref.grad is None and advantages.grad is None
    assert logp.grad[0,1,1]==0
    assert info['kl']>=0


def test_gradient_direction_and_zero_advantage_kl():
    x=torch.zeros(1,2,1,dtype=torch.float64,requires_grad=True)
    a=group_advantages(torch.tensor([[0.,1.]],dtype=torch.float64))
    mask=torch.ones_like(x,dtype=torch.bool)
    grpo_loss(x,x.detach(),x.detach(),a,mask,beta=0)[0].backward()
    assert x.grad[0,0]>0 and x.grad[0,1]<0
    a=group_advantages(torch.ones(1,2))
    assert a.eq(0).all() and torch.isfinite(a).all()
    x=torch.full((1,2,1),-.5,dtype=torch.float64,requires_grad=True)
    loss,_=grpo_loss(x,x.detach(),torch.full_like(x,-1.),a,mask)
    loss.backward()
    assert x.grad.abs().sum()>0  # zero advantage does not remove KL


def test_eos_padding_and_masked_nonfinite_values():
    tokens=torch.tensor([[3,1,0,0],[1,0,0,0],[3,4,0,0]])
    mask=response_mask(tokens,eos=1,lengths=torch.tensor([2,1,2]))
    assert mask.tolist()==[[True,True,False,False],[True,False,False,False],[True,True,False,False]]
    x=torch.zeros(1,3,4,dtype=torch.float64,requires_grad=True)
    padded=x.masked_fill(~mask[None],float('nan'))
    loss,_=grpo_loss(padded,padded,padded,torch.zeros(1,3),mask[None])
    assert loss==0


def test_frozen_snapshots_on_policy_and_response_logp():
    torch.manual_seed(12)
    model=TinyLM(vocab=8,width=8,heads=2)
    old,reference=frozen_copy(model),frozen_copy(model)
    snapshot={k:v.clone() for k,v in old.state_dict().items()}
    prompts=torch.tensor([[2,3],[2,4]])
    response,mask,logp=sample_completions(old,prompts,3,3,1,0,torch.Generator().manual_seed(5))
    expanded=prompts.repeat_interleave(3,0)
    new=completion_logp(model,expanded,response.flatten(0,1)).reshape_as(logp)
    torch.testing.assert_close(logp,new)
    joined=torch.cat((expanded,response.flatten(0,1)),-1)
    for i in range(3):
        oracle=old(joined[:,:2+i])[:,-1].log_softmax(-1).gather(1,joined[:,2+i,None]).squeeze(-1)
        torch.testing.assert_close(logp.flatten(0,1)[:,i],oracle,atol=1e-6,rtol=1e-6)
    adv=group_advantages(torch.tensor([[0.,1.,0.],[1.,0.,1.]]))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    grpo_loss(new,logp,completion_logp(reference,expanded,response.flatten(0,1)).reshape_as(logp),adv,mask)[0].backward()
    optimizer.step()
    assert not logp.requires_grad
    for frozen in (old,reference):
        assert all(p.grad is None and not p.requires_grad for p in frozen.parameters())
        for k,v in frozen.state_dict().items(): torch.testing.assert_close(v,snapshot[k])


def test_arithmetic_sft_tiny_batch_overfit():
    torch.manual_seed(17)
    policy=TinyLM(vocab=24,width=16,heads=2)
    prompts=torch.tensor([[20,1,21,2,22],[20,2,21,4,22],[20,4,21,3,22],[20,5,21,1,22]])
    target=torch.tensor([[3,19],[6,19],[7,19],[6,19]])
    optimizer=torch.optim.Adam(policy.parameters(),lr=.02)
    for _ in range(65):
        optimizer.zero_grad()
        loss=-completion_logp(policy,prompts,target).mean()
        loss.backward(); optimizer.step()
    assert loss.item()<.02
    policy.eval()
    responses,mask,_=sample_completions(policy,prompts,1,3,19,23,torch.Generator().manual_seed(7),greedy=True)
    assert torch.equal(responses[:,0,:2],target)
    assert mask.sum(-1).eq(2).all()
