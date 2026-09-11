"""Generated regression: architectural comparison or paired shift investigation."""
import argparse
import time
import torch
from deepseek_minimal.moe import MoE, balance_losses
from examples.run_utils import setup, metadata, memory, timing, save, restore, write_json


def task(generator,batch,probability):
    x=torch.randn(batch,32,generator=generator)*.5
    cluster=torch.rand(batch,generator=generator)<probability
    x[:,0]=torch.where(cluster,1.,-1.)
    target=torch.tanh(x.roll(1,-1)*.7+x.roll(2,-1)*x[:,0,None]*.4)
    return x,target


def adaptation(rows,shift):
    rolling=[sum(r['imbalance'] for r in rows[i-4:i+1])/5 for i in range(shift+4,len(rows))]
    for i in range(2,len(rolling)):
        if max(rolling[i-2:i+1])<=.3:
            return i+5
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['fine','coarse','instant','ema'],default='fine')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--output')
    args=parser.parse_args()
    if not 10<=args.steps<=2000: parser.error('Bounded steps must be 10..2000')
    config=dict(variant=args.variant,seed=args.seed,steps=args.steps,batch=64,width=32,
                learning_rate=.003,shift=args.steps//2,wall_budget_seconds=120)
    setup(args.seed)
    if args.variant=='coarse':
        model=MoE(hidden=32,routed=4,shared=0,top_k=2)
    else:
        model=MoE(score='sigmoid' if args.variant in ('instant','ema') else 'softmax',
                  bias_rate=.001 if args.variant in ('instant','ema') else 0.,
                  ema_decay=.9 if args.variant=='ema' else 0.)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    generator=torch.Generator().manual_seed(args.seed+1000)
    checkpoint=f'artifacts/moe-{args.variant}-{args.seed}/checkpoint.pt'
    start,rows=0,[]
    if args.resume:
        start,extra=restore(args.resume,model,optimizer,generator,config); rows=extra['rows']
    initial_memory=memory(model,optimizer)
    run_start=time.perf_counter()
    for step in range(start,min(args.steps,args.stop_after or args.steps)):
        tick=time.perf_counter()
        x,y=task(generator,64,.85 if step<config['shift'] else .15)
        optimizer.zero_grad()
        predicted,info=model(x)
        task_loss=(predicted-y).square().mean()
        auxiliary=balance_losses(info['scores'],info['indices'])[0] if args.variant in ('fine','coarse') else task_loss*0
        (task_loss+.001*auxiliary).backward(); optimizer.step()
        old_bias=model.selection_bias.clone()
        controller_start=time.perf_counter(); model.finish_step(); controller_time=time.perf_counter()-controller_start
        loads=info['loads'].tolist()
        rows.append(dict(step=step,loss=task_loss.item(),loads=loads,
                         imbalance=max(loads)/(sum(loads)/len(loads))-1,
                         bias_delta=(model.selection_bias-old_bias).tolist(),
                         seconds=time.perf_counter()-tick,controller_seconds=controller_time))
        if step==start: initial_memory['after_first_step']=memory(model,optimizer)
        if time.perf_counter()-run_start>config['wall_budget_seconds']:
            save(checkpoint,model,optimizer,config,step+1,generator,dict(rows=rows))
            raise RuntimeError('Wall budget reached; partial checkpoint saved')
    save(checkpoint,model,optimizer,config,len(rows),generator,dict(rows=rows))
    if len(rows)<args.steps:
        print('Partial checkpoint:',checkpoint); return
    model.eval()
    with torch.no_grad():
        x,y=task(torch.Generator().manual_seed(args.seed+90000),512,.15)
        predicted,info=model(x)
        heldout=(predicted-y).square().mean().item()
    late=rows[-20:]
    switches=[]
    for a,b in zip(late,late[1:]):
        switches.append(sum(x*y<0 for x,y in zip(a['bias_delta'],b['bias_delta']))/len(a['loads']))
    result=metadata(config)
    result.update(parameters=model.parameter_counts(),memory=memory(model,optimizer),initial_memory=initial_memory,
                  task='two-cluster continuous regression; held-out RNG seed+90000',heldout_mse=heldout,
                  heldout_loads=info['loads'].tolist(),late_imbalance=sum(r['imbalance'] for r in late)/len(late),
                  switch_rate=sum(switches)/len(switches),adaptation_updates=adaptation(rows,config['shift']),
                  step_time=timing([r['seconds'] for r in rows[5:]]),
                  controller_time=timing([r['controller_seconds'] for r in rows[5:]]),
                  pre_shift_imbalance=sum(r['imbalance'] for r in rows[config['shift']-20:config['shift']])/20,
                  input_tokens=len(rows)*64,checkpoint=checkpoint,
                  log_path=f'artifacts/moe-{args.variant}-{args.seed}/log.json')
    write_json(result['log_path'],rows)
    output=args.output or f'results/moe-{args.variant}-{args.seed}.json'
    write_json(output,result)
    print(output,'heldout_mse=',heldout)


if __name__=='__main__': main()
