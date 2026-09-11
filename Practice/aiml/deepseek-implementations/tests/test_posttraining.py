import io
import pytest
import torch
from deepseek_minimal.models import r1
from deepseek_minimal.grpo import frozen_copy, response_mask
from deepseek_minimal.posttraining import supervised_step, rejection_data, reinforcement_step
from deepseek_minimal.rotary import inverse_frequencies


@pytest.mark.parametrize('family',['qwen2','qwen3','llama3'])
def test_student_cache_and_serialization(family):
    model = r1.build_student(family).double().eval()
    tokens = torch.tensor([[1,2,3,4,5]])
    a,cache = model.decode(tokens[:,:2])
    b,_ = model.decode(tokens[:,2:],cache)
    torch.testing.assert_close(torch.cat((a,b),1),model(tokens))
    stream = io.BytesIO()
    torch.save(model.state_dict(),stream); stream.seek(0)
    other = r1.build_student(family).double().eval()
    other.load_state_dict(torch.load(stream,weights_only=True))
    torch.testing.assert_close(other(tokens),model(tokens))


def test_wavelength_scaling_matches_piecewise_scalar_rule():
    plain = inverse_frequencies(128,base=500000.,dtype=torch.float64)
    expected = []
    for frequency in plain.tolist():
        rotations = 8192*frequency/(2*torch.pi)
        if rotations < 1:
            expected.append(frequency/8)
        elif rotations > 4:
            expected.append(frequency)
        else:
            blend = (rotations-1)/3
            expected.append(frequency*blend+frequency/8*(1-blend))
    actual = inverse_frequencies(128,base=500000.,factor=8,scheme='llama3',original_length=8192,dtype=torch.float64)
    torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float64))


def test_verified_rejection_and_empty_selection():
    prompts = torch.tensor([[1,2],[3,4]])
    responses = torch.tensor([[[5,9,0],[5,9,0],[6,9,0]],[[7,9,0],[8,9,0],[0,0,0]]])
    mask = response_mask(responses,9)
    good = torch.tensor([[True,True,False],[True,False,False]])
    p,r,lengths = rejection_data(prompts,responses,mask,good)
    assert p.tolist() == [[1,2],[3,4]] and r.tolist() == [[5,9,0],[7,9,0]]
    assert lengths.tolist() == [2,2]
    empty = rejection_data(prompts,responses,mask,torch.zeros_like(good))
    assert all(t.shape[0] == 0 for t in empty)


def test_supervised_to_rl_snapshot_and_reference_immutability():
    torch.manual_seed(8)
    model = r1.build(vocab=12)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002)
    prompts,responses = torch.tensor([[1,2],[3,4]]),torch.tensor([[5,9],[7,9]])
    supervised_step(model,optimizer,prompts,responses,9)
    reference = frozen_copy(model)  # Live routing diagnostics must not break deepcopy.
    before = {n:p.clone() for n,p in reference.state_dict().items()}
    info = reinforcement_step(model,reference,optimizer,prompts,
                              lambda r,m:r[...,0].eq(5).float(),torch.Generator().manual_seed(2),9,0)
    assert torch.isfinite(torch.tensor(info['loss']))
    for n,p in reference.state_dict().items():
        assert torch.equal(before[n],p)
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())
