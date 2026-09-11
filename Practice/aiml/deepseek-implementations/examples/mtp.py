"""Paired NTP versus NTP+MTP on generated modular progressions."""
import argparse
import time
import torch
from deepseek_minimal.mtp import MTPModel, prediction_losses
from examples.run_utils import setup, metadata, memory, timing, save, restore, write_json


def sequences(generator,batch):
    start=torch.randint(0,16,(batch,1),generator=generator)
    stride=torch.randint(1,5,(batch,1),generator=generator)
    return (start+stride*torch.arange(16))%16


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['ntp','mtp'],default='mtp')
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--output')
    args=parser.parse_args()
    if not 1<=args.steps<=1000: parser.error('steps must be 1..1000')
    config=dict(variant=args.variant,seed=args.seed,steps=args.steps,batch=16,width=32,
                length=16,lr=.003,mtp_weight=.3,wall_budget_seconds=120)
    setup(args.seed)
    model=MTPModel(); optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    generator=torch.Generator().manual_seed(args.seed+1000)
    lengths=torch.full((16,),16)
    start,rows=0,[]
    if args.resume: start,extra=restore(args.resume,model,optimizer,generator,config); rows=extra['rows']
    initial_memory=memory(model)
    clock=time.perf_counter()
    for step in range(start,min(args.steps,args.stop_after or args.steps)):
        tick=time.perf_counter(); tokens=sequences(generator,16)
        optimizer.zero_grad()
        a,b=prediction_losses(*model(tokens,include_mtp=args.variant=='mtp'),tokens,lengths)
        (a+.3*b).backward(); optimizer.step()
        rows.append(dict(ntp_loss=a.item(),mtp_loss=b.item(),seconds=time.perf_counter()-tick))
        if time.perf_counter()-clock>120:
            save(f'artifacts/mtp-{args.variant}/checkpoint.pt',model,optimizer,config,step+1,generator,dict(rows=rows))
            raise RuntimeError('Training wall budget exceeded; partial checkpoint saved')
    checkpoint=f'artifacts/mtp-{args.variant}/checkpoint.pt'
    save(checkpoint,model,optimizer,config,len(rows),generator,dict(rows=rows))
    if len(rows)<args.steps: print('Partial checkpoint:',checkpoint); return
    model.eval()
    # Exhaust all 64 (start,stride) combinations. Combinations occur in training;
    # this tests the generated distribution, not compositional generalization.
    tokens=torch.stack([(a+b*torch.arange(16))%16 for a in range(16) for b in range(1,5)])
    with torch.no_grad():
        logits=model(tokens,False)[0]
        loss,_=prediction_losses(logits,None,tokens,torch.full((64,),16))
        accuracy=logits[:,1:-1].argmax(-1).eq(tokens[:,2:]).float().mean().item()
    result=metadata(config)
    base_params=sum(p.numel() for p in model.base.parameters())
    result.update(parameters_total=sum(p.numel() for p in model.parameters()),base_parameters=base_params,
                  parameters_used=base_params if args.variant=='ntp' else sum(p.numel() for p in model.parameters()),
                  initial_memory=initial_memory,memory=memory(model,optimizer),eval_ntp_loss=loss.item(),
                  eval_predictable_accuracy=accuracy,task='mod-16 progressions, stride 1..4; all 64 combinations evaluated',
                  step_time=timing([r['seconds'] for r in rows[5:]] or [r['seconds'] for r in rows]),
                  base_token_positions=len(rows)*16*16,extra_block_positions=len(rows)*16*15 if args.variant=='mtp' else 0,
                  checkpoint=checkpoint,log_path=f'artifacts/mtp-{args.variant}/log.json')
    write_json(result['log_path'],rows)
    output=args.output or f'results/mtp-{args.variant}.json'
    write_json(output,result); print(output,'accuracy=',accuracy)


if __name__=='__main__': main()
