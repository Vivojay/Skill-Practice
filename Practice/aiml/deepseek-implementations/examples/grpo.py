"""Math-v1 outcome GRPO after tiny SFT on generated integer addition."""
import argparse
import time
import torch
from deepseek_minimal.mtp import TinyLM
from deepseek_minimal.grpo import (frozen_copy, sample_completions, completion_logp,
                                   group_advantages, grpo_loss)
from examples.run_utils import setup, metadata, memory, timing, save, restore, write_json

EOS, BOS, PLUS, EQUAL, PAD = 19,20,21,22,23


def data():
    pairs=torch.cartesian_prod(torch.arange(10),torch.arange(10))
    prompts=torch.stack((torch.full((100,),BOS),pairs[:,0],torch.full((100,),PLUS),pairs[:,1],torch.full((100,),EQUAL)),-1)
    answers=pairs.sum(-1)
    heldout=(pairs[:,0]+3*pairs[:,1])%5==0
    return prompts,answers,heldout


def rewards(responses,mask,answers):
    return (responses[...,0].eq(answers[:,None]) & responses[...,1].eq(EOS) & mask.sum(-1).eq(2)).float()


@torch.no_grad()
def evaluate(policy,reference,prompts,answers,seed):
    sampled,mask,logp=sample_completions(policy,prompts,16,3,EOS,PAD,torch.Generator().manual_seed(seed))
    reward=rewards(sampled,mask,answers)
    reference_logp=completion_logp(reference,prompts.repeat_interleave(16,0),sampled.flatten(0,1)).reshape_as(logp)
    _,info=grpo_loss(logp,logp,reference_logp,torch.zeros_like(reward),mask)
    greedy,gm,_=sample_completions(policy,prompts,1,3,EOS,PAD,torch.Generator().manual_seed(seed),greedy=True)
    return dict(sampled_exact_correctness=reward.mean().item(),greedy_exact_correctness=rewards(greedy,gm,answers).mean().item(),
                mean_length=mask.sum(-1).float().mean().item(),reward_dispersion=reward.std(-1,correction=0).mean().item(),
                kl_k3=info['kl'].item(),questions=len(prompts),samples_per_question=16)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--sft-steps',type=int,default=160)
    parser.add_argument('--rl-steps',type=int,default=60)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--output',default='results/grpo.json')
    args=parser.parse_args()
    if not 1<=args.sft_steps<=1000 or not 1<=args.rl_steps<=300: parser.error('SFT 1..1000; RL 1..300')
    config=dict(seed=args.seed,sft_steps=args.sft_steps,rl_steps=args.rl_steps,width=32,
                sft_batch=32,rl_questions=8,group=8,response_cap=3,mu=2,beta=.04,clip=.2,
                sft_lr=.003,rl_lr=.0005,wall_budget_seconds=180,paper='2402.03300v1')
    setup(args.seed)
    policy=TinyLM(vocab=24)
    optimizer=torch.optim.Adam(policy.parameters(),lr=config['sft_lr'])
    generator=torch.Generator().manual_seed(args.seed+1000)
    prompts,answers,heldout=data()
    train,truth=prompts[~heldout],answers[~heldout]
    start,rows,reference=0,[],None
    if args.resume:
        start,extra=restore(args.resume,policy,optimizer,generator,config); rows=extra['rows']
        if extra['reference'] is not None:
            reference=frozen_copy(policy); reference.load_state_dict(extra['reference'])
    initial_memory=memory(policy)
    clock=time.perf_counter()
    total=args.sft_steps+args.rl_steps
    checkpoint='artifacts/grpo/checkpoint.pt'
    for step in range(start,min(total,args.stop_after or total)):
        tick=time.perf_counter()
        if step<args.sft_steps:
            indices=torch.randint(len(train),(32,),generator=generator)
            response=torch.stack((truth[indices],torch.full((32,),EOS)),-1)
            optimizer.zero_grad()
            loss=-completion_logp(policy,train[indices],response).mean()
            loss.backward(); optimizer.step()
            row=dict(phase='sft',loss=loss.item())
        else:
            if reference is None: reference=frozen_copy(policy)
            for group in optimizer.param_groups: group['lr']=config['rl_lr']
            old=frozen_copy(policy)
            indices=torch.randint(len(train),(8,),generator=generator)
            batch=train[indices]
            response,mask,old_logp=sample_completions(old,batch,8,3,EOS,PAD,generator)
            reward=rewards(response,mask,truth[indices])
            advantage=group_advantages(reward)
            expanded=batch.repeat_interleave(8,0)
            with torch.no_grad(): ref_logp=completion_logp(reference,expanded,response.flatten(0,1)).reshape_as(old_logp)
            for _ in range(config['mu']):
                optimizer.zero_grad()
                new_logp=completion_logp(policy,expanded,response.flatten(0,1)).reshape_as(old_logp)
                loss,info=grpo_loss(new_logp,old_logp,ref_logp,advantage,mask,clip=.2,beta=.04)
                loss.backward(); optimizer.step()
            row=dict(phase='grpo',loss=loss.item(),reward=reward.mean().item(),
                     dispersion=reward.std(-1,correction=0).mean().item(),kl=info['kl'].item(),
                     length=mask.sum(-1).float().mean().item(),zero_advantage_groups=advantage.eq(0).all(-1).sum().item())
        row['seconds']=time.perf_counter()-tick; rows.append(row)
        if time.perf_counter()-clock>180:
            save(checkpoint,policy,optimizer,config,step+1,generator,dict(rows=rows,reference=reference.state_dict() if reference else None))
            raise RuntimeError('Wall budget exceeded; partial checkpoint saved')
    save(checkpoint,policy,optimizer,config,len(rows),generator,dict(rows=rows,reference=reference.state_dict() if reference else None))
    if len(rows)<total: print('Partial checkpoint:',checkpoint); return
    policy.eval()
    result=metadata(config)
    result.update(parameters=sum(p.numel() for p in policy.parameters()),initial_memory=initial_memory,memory=memory(policy,optimizer),
                  task='integer addition; answer is one sum token followed by EOS, no reasoning text',
                  split='80 train / 20 held-out operand pairs, held-out iff (a+3*b)%5==0',
                  sft_train=evaluate(reference,reference,train,truth,args.seed+90000),
                  sft_heldout=evaluate(reference,reference,prompts[heldout],answers[heldout],args.seed+90001),
                  grpo_heldout=evaluate(policy,reference,prompts[heldout],answers[heldout],args.seed+90001),
                  sft_timing=timing([r['seconds'] for r in rows if r['phase']=='sft']),
                  rl_timing=timing([r['seconds'] for r in rows if r['phase']=='grpo']),
                  checkpoint=checkpoint,log_path='artifacts/grpo/log.json',
                  generated_rl_completions=args.rl_steps*8*8,rl_optimizer_steps=args.rl_steps*2)
    write_json(result['log_path'],rows); write_json(args.output,result)
    print(args.output,'SFT train correctness',result['sft_train']['greedy_exact_correctness'],
          'heldout SFT/GRPO',result['sft_heldout']['greedy_exact_correctness'],result['grpo_heldout']['greedy_exact_correctness'])


if __name__=='__main__': main()
