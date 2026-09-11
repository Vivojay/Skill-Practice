"""Tiny attention overfit plus separately timed prefill and one-token decode."""
import argparse
import time
import torch
from deepseek_minimal.mla import MLA
from examples.run_utils import setup, metadata, memory, timing, save, restore, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--output',default='results/mla.json')
    args=parser.parse_args()
    if not 1<=args.steps<=1000: parser.error('steps must be 1..1000')
    config=dict(seed=args.seed,steps=args.steps,width=32,heads=4,kv_rank=8,q_rank=16,
                batch=2,length=16,lr=.01,wall_budget_seconds=120)
    setup(args.seed)
    model=MLA()
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    generator=torch.Generator().manual_seed(args.seed+1000)
    x=torch.randn(2,8,32,generator=generator)
    target=x.roll(1,1); target[:,0]=0
    first=(model(x)-target).square().mean().item()
    start,history=0,[]
    if args.resume:
        start,extra=restore(args.resume,model,optimizer,generator,config)
        x,target,first,history=extra['x'],extra['target'],extra['first'],extra['history']
    initial_memory=memory(model)
    clock=time.perf_counter()
    for step in range(start,min(args.steps,args.stop_after or args.steps)):
        tick=time.perf_counter(); optimizer.zero_grad()
        loss=(model(x)-target).square().mean()
        loss.backward(); optimizer.step()
        history.append(dict(loss=loss.item(),seconds=time.perf_counter()-tick))
        if time.perf_counter()-clock>120:
            save('artifacts/mla/checkpoint.pt',model,optimizer,config,step+1,generator,
                 dict(x=x,target=target,first=first,history=history))
            raise RuntimeError('Training wall budget exceeded; partial checkpoint saved')
    checkpoint='artifacts/mla/checkpoint.pt'
    save(checkpoint,model,optimizer,config,len(history),generator,dict(x=x,target=target,first=first,history=history))
    if len(history)<args.steps: print('Partial checkpoint:',checkpoint); return
    model.eval()
    with torch.no_grad(): final=(model(x)-target).square().mean().item()
    benchmark=torch.randn(2,16,32,generator=generator)
    result=metadata(config)
    result.update(first_mse=first,final_mse=final,parameters=sum(p.numel() for p in model.parameters()),
                  initial_memory=initial_memory,memory=memory(model,optimizer),train_time=timing([r['seconds'] for r in history]),
                  checkpoint=checkpoint,task='fixed tiny batch: predict preceding input vector; not held-out quality')
    for expanded in [False,True]:
        label='expanded' if expanded else 'compressed'
        reference,cache,_=model.decode(benchmark,expanded=expanded)
        _,prefix,_=model.decode(benchmark[:,:-1],expanded=expanded)
        measurements={}
        for phase in ['prefill','decode']:
            times=[]
            for i in range(30):
                tick=time.perf_counter()
                if phase=='prefill': model.decode(benchmark,expanded=expanded)
                else: model.decode(benchmark[:,-1:],prefix,expanded=expanded)
                if i>=5: times.append(time.perf_counter()-tick)
            measurements[phase]=timing(times)
        result[label]=dict(cache_bytes=cache.nbytes,**measurements)
    expanded=model.decode(benchmark,expanded=True)[0]
    compressed=model.decode(benchmark)[0]
    result['max_output_error']=(expanded-compressed).abs().max().item()
    write_json(args.output,result)
    print(args.output,'MSE',first,'->',final,'cache bytes',result['compressed']['cache_bytes'])


if __name__=='__main__': main()
