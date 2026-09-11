import torch
from deepseek_minimal.moe import MoE, balance_losses


def test_explicit_expert_sum_and_gradients():
    torch.manual_seed(7)
    model = MoE(width=4, hidden=3, routed=5, shared=1, top_k=2).double()
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    actual, info = model(x)
    expected = []
    for token in x.reshape(-1, 4):
        probabilities = torch.softmax(model.router.weight @ token, 0)
        selected = sorted(range(5), key=lambda i: float(probabilities[i].detach()), reverse=True)[:2]
        expected.append(model.shared[0](token) + sum(probabilities[i] * model.experts[i](token) for i in selected))
    expected = torch.stack(expected).reshape_as(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    assert info['loads'].sum() == 12
    for i, expert in enumerate(model.experts):
        if info['loads'][i]:
            assert expert.down.weight.grad.abs().sum() > 0


def test_empty_experts_and_unbiased_weights():
    model = MoE(width=2, hidden=3, routed=4, shared=0, top_k=1, score='sigmoid').double()
    model.selection_bias[2] = 100
    x = torch.ones(3, 2, dtype=torch.float64)
    y, info = model(x)
    assert info['indices'].eq(2).all()
    torch.testing.assert_close(info['weights'][:, 0], (x @ model.router.weight[2]).sigmoid())
    y.sum().backward()
    assert model.experts[0].down.weight.grad is None
    assert model.experts[2].down.weight.grad is not None


def test_balance_objectives_manual():
    scores = torch.tensor([[.4,.3,.2,.1], [.6,.1,.1,.2]], dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[0,1],[0,3]])
    expert, device = balance_losses(scores, indices, 2)
    # f=[2,1,0,1], P=[.5,.2,.15,.15]; group f=[1.5,.5], P=[.7,.3]
    torch.testing.assert_close(expert, torch.tensor(1.35, dtype=torch.float64))
    torch.testing.assert_close(device, torch.tensor(1.2, dtype=torch.float64))
    expert.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([[1.,.5,0.,.5]]*2, dtype=torch.float64))


def test_training_boundary_accumulation_and_eval_freeze():
    model = MoE(width=2, routed=3, top_k=1, bias_rate=.1)
    model.selection_bias[0] = 10
    before = model.selection_bias.clone()
    model(torch.ones(2, 2)); model(torch.ones(3, 2))
    assert model.pending_load.tolist() == [5,0,0]
    torch.testing.assert_close(before, model.selection_bias)
    model.finish_step()
    torch.testing.assert_close(model.selection_bias - before, torch.tensor([-.1,.1,.1]))
    assert model.pending_load.sum() == 0
    snapshot = {k:v.clone() for k,v in model.state_dict().items()}
    model.eval(); model(torch.randn(10,2)); model.finish_step()
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v, snapshot[k])


def test_tiny_batch_overfit():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = MoE(width=4, hidden=8, routed=3, shared=1, top_k=1)
    x, target = torch.randn(8,4), torch.randn(8,4)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    first = (model(x)[0]-target).square().mean().item()
    for _ in range(120):
        optimizer.zero_grad()
        loss = (model(x)[0]-target).square().mean()
        loss.backward(); optimizer.step()
    assert loss.item() < first * .03


def test_instantaneous_exact_tie_and_ema_history():
    model=MoE(routed=7,bias_rate=.01)
    model.pending_load.copy_(torch.tensor([3.,0.,0.,2.,4.,5.,7.]))
    model.finish_step()
    torch.testing.assert_close(model.selection_bias,torch.tensor([0.,.01,.01,.01,-.01,-.01,-.01]))
    ema=MoE(routed=3,top_k=1,bias_rate=.01,ema_decay=.9)
    ema.pending_load.copy_(torch.tensor([10.,0.,0.])); ema.finish_step()
    ema.pending_load.copy_(torch.tensor([0.,0.,10.])); ema.finish_step()
    torch.testing.assert_close(ema.ema_load,torch.tensor([.9,0.,.1]))
    # Still downweights expert zero because the EMA retains history.
    torch.testing.assert_close(ema.selection_bias,torch.tensor([-.02,.02,.02]))
