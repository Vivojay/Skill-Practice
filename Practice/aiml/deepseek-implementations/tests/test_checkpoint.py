import torch
from deepseek_minimal.moe import MoE
from examples.run_utils import setup, save, restore


def test_optimizer_rng_task_and_controller_resume_exactly():
    setup(123)
    model=MoE(width=4,hidden=4,routed=3,top_k=1,bias_rate=.001,ema_decay=.9)
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    generator=torch.Generator().manual_seed(42)
    def step():
        optimizer.zero_grad()
        x=torch.randn(4,4,generator=generator)+torch.rand(1)
        model(x)[0].square().mean().backward(); optimizer.step(); model.finish_step()
    step()
    # BytesIO avoids platform temp-directory permissions and still exercises serialization.
    import io
    buffer=io.BytesIO()
    config=dict(seed=123)
    save(buffer,model,optimizer,config,1,generator)
    step()
    expected={k:v.clone() for k,v in model.state_dict().items()}
    buffer.seek(0)
    restored_step,_=restore(buffer,model,optimizer,generator,config)
    assert restored_step==1
    step()
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
