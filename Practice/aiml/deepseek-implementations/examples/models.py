"""Train a small family decoder on generated modular progressions."""
import argparse
import time
import torch
from torch.nn import functional as F
from deepseek_minimal.models import v1,v2,v3,v32,v4,r1
from deepseek_minimal.models.prediction import MultiTokenModel
from deepseek_minimal.dsa import SparseMLA
from deepseek_minimal.moe import MoE
from deepseek_minimal.mtp import prediction_losses
from examples.run_utils import setup,metadata,memory,timing,save,restore,write_json


FAMILIES = ('v1','moe','v2','v3','v32','v4','v3-mtp','v4-mtp','qwen2','qwen3','llama3')


def build(family):
    if family == 'v1': return v1.build(vocab=16)
    if family == 'moe': return v1.build(vocab=16,moe=True)
    if family == 'v2': return v2.build(vocab=16)
    if family == 'v3': return v3.build(vocab=16)
    if family == 'v32': return v32.build(vocab=16)
    if family == 'v4': return v4.build(vocab=16)
    if family == 'v3-mtp': return MultiTokenModel(v3.build(vocab=16))
    if family == 'v4-mtp': return v4.MultiTokenModel(v4.build(vocab=16))
    return r1.build_student(family,vocab=16)


def phase(model,step,warmup):
    model.train().requires_grad_(True)
    sparse = [m for m in model.modules() if isinstance(m,SparseMLA)]
    for m in sparse: m.dense_warmup = step < warmup
    if sparse and step < warmup:
        model.requires_grad_(False)
        for m in sparse: m.indexer.requires_grad_(True)
        for m in model.modules():
            if isinstance(m,MoE): m.eval()
    return bool(sparse and step < warmup)


def sequences(generator,batch=8):
    start = torch.randint(0,12,(batch,1),generator=generator)
    stride = torch.randint(1,4,(batch,1),generator=generator)
    return (start+stride*torch.arange(12))%16


@torch.no_grad()
def evaluate(model,tokens):
    model.eval()
    logits = model(tokens)
    loss = F.cross_entropy(logits[:,:-1].transpose(1,2),tokens[:,1:])
    # First transition does not reveal the stride; exclude it from predictable accuracy.
    accuracy = logits[:,1:-1].argmax(-1).eq(tokens[:,2:]).float().mean()
    return dict(loss=loss.item(),predictable_accuracy=accuracy.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=FAMILIES,default='v3')
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not 1 <= args.steps <= 500: parser.error('Use 1..500 steps')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps: parser.error('Invalid stop boundary')
    config = dict(family=args.family,steps=args.steps,seed=args.seed,width=32,layers=2,
                  batch=8,length=12,vocab=16,lr=.003,warmup=8 if args.family=='v32' else 0,
                  mtp_weight=.3,balance_weight=.001,index_weight=.1,wall_budget_seconds=180)
    setup(args.seed)
    model = build(args.family)
    base = model.base if hasattr(model,'base') else model
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'])
    generator = torch.Generator().manual_seed(args.seed+1000)
    validation = torch.stack([(a+b*torch.arange(12))%16 for a in range(12,16) for b in range(1,4)])
    rows,start = [],0
    initial = evaluate(base,validation)
    if args.resume:
        start,extra = restore(args.resume,model,optimizer,generator,config)
        rows,initial = extra['rows'],extra['initial']
    checkpoint = f'artifacts/models-{args.family}-{args.seed}/checkpoint.pt'
    clock = time.perf_counter()
    for step in range(start,args.stop_after or args.steps):
        tick = time.perf_counter()
        warmup = phase(model,step,config['warmup'])
        tokens = sequences(generator)
        optimizer.zero_grad()
        output = model(tokens)
        ntp,mtp = output if isinstance(output,tuple) else (output,None)
        a,b = prediction_losses(ntp,mtp,tokens,torch.full((8,),12))
        index_losses = [m.index_objective for m in model.modules()
                        if getattr(m,'index_objective',None) is not None]
        index = torch.stack(index_losses).mean() if index_losses else a*0
        loss = index if warmup else a+.3*b+.001*base.balance_loss()+.1*index
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step(); model.finish_step()
        rows.append(dict(step=step+1,phase='indexer_warmup' if warmup else 'train',
                         ntp_loss=a.item(),mtp_loss=b.item(),index_loss=index.item(),
                         seconds=time.perf_counter()-tick))
        if time.perf_counter()-clock > config['wall_budget_seconds']:
            save(checkpoint,model,optimizer,config,step+1,generator,dict(rows=rows,initial=initial))
            raise RuntimeError('Wall budget reached; partial checkpoint saved')
    save(checkpoint,model,optimizer,config,len(rows),generator,dict(rows=rows,initial=initial))
    if len(rows) < args.steps:
        print('Partial checkpoint:',checkpoint); return
    final = evaluate(base,validation)
    prompt = validation[:1,:4]
    generated = base.generate(prompt,5)
    _,caches = base.decode(validation[:1])
    result = metadata(config)
    result.update(initial=initial,final=final,parameters=sum(p.numel() for p in model.parameters()),
                  memory=memory(model,optimizer),step_time=timing([r['seconds'] for r in rows]),
                  cache_bytes_at_12_tokens=sum(c.nbytes for c in caches),
                  prompt=prompt.tolist(),generated=generated.tolist(),checkpoint=checkpoint,
                  task='mod-16 progressions, stride 1..3; training starts 0..11, validation starts 12..15',
                  limitation='Same arithmetic rule and overlapping subsequences; not a general reasoning benchmark',
                  first_training_loss=rows[0]['ntp_loss'],last_training_loss=rows[-1]['ntp_loss'])
    write_json(checkpoint.replace('checkpoint.pt','log.json'),rows)
    output = args.output or f'results/model-{args.family}.json'
    write_json(output,result)
    print(args.family,initial,final,'cache bytes',result['cache_bytes_at_12_tokens'])


if __name__ == '__main__': main()
