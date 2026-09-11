"""Bounded R1 pipeline analogue on generated addition and echo tasks.

No reasoning traces or human preference model are supplied. The Zero path skips
answer SFT after a generic sequence pretraining stage; it is not R1-Zero itself.
Checkpoints resume at completed stage boundaries.
"""
import argparse
import copy
import time
import torch
from torch.nn import functional as F
from deepseek_minimal.models import r1
from deepseek_minimal.grpo import frozen_copy,sample_completions
from deepseek_minimal.posttraining import supervised_step,reinforcement_step,rejection_data
from examples.grpo import data,rewards,evaluate,EOS,PAD
from examples.run_utils import setup,metadata,save,restore,write_json


def pretrain(model,optimizer,generator,steps,check_time):
    model.train()
    for _ in range(steps):
        start = torch.randint(0,19,(16,1),generator=generator)
        stride = torch.randint(1,4,(16,1),generator=generator)
        tokens = (start+stride*torch.arange(8))%19
        optimizer.zero_grad()
        loss = F.cross_entropy(model(tokens)[:,:-1].transpose(1,2),tokens[:,1:])
        loss.backward(); optimizer.step(); model.finish_step()
        check_time()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',choices=['r1','zero'],default='r1')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-stage',type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    config = dict(recipe=args.recipe,seed=args.seed,base_steps=30,cold_steps=120,rl_steps=20,
                  reset_sft_steps=60,student_steps=100,student_base_steps=20,
                  group=8,max_response=3,updates_per_group=2,wall_budget_seconds=240,
                  objective='DeepSeekMath v1 token-level outcome GRPO; R1 stage-order analogue')
    stages = ['base_pretrain','reasoning_rl'] if args.recipe=='zero' else [
        'base_pretrain','cold_sft','reasoning_rl','rejection','reset_mixed_sft',
        'mixed_rl','student_qwen2','student_qwen3','student_llama3']
    if args.stop_after_stage is not None and not 1 <= args.stop_after_stage <= len(stages):
        parser.error('Invalid stage boundary')
    setup(args.seed)
    policy = r1.build(vocab=24)
    optimizer = torch.optim.AdamW(policy.parameters(),lr=.003)
    generator = torch.Generator().manual_seed(args.seed+1000)
    prompts,answers,heldout = data()
    train,truth = prompts[~heldout],answers[~heldout]
    target = torch.stack((truth,torch.full_like(truth,EOS),torch.full_like(truth,PAD)),-1)
    # A separate copy/echo domain makes the mixed-data stage observable.
    echo = train.clone(); echo[:,2] = 22
    echo_target = target.clone(); echo_target[:,0] = echo[:,1]
    extra = dict(rows=[],base_state=None,rejected_prompts=train[:0],rejected_responses=target[:0],
                 rejected_lengths=truth[:0],students={})
    start = 0
    if args.resume:
        start,extra = restore(args.resume,policy,optimizer,generator,config)
    checkpoint = f'artifacts/reasoning-{args.recipe}-{args.seed}/checkpoint.pt'
    clock = time.perf_counter()
    def check_time():
        if time.perf_counter()-clock > config['wall_budget_seconds']:
            raise RuntimeError('Wall budget reached; resume the last completed stage checkpoint')
    # Initial boundary is recoverable even if the first stage exhausts its budget.
    if not args.resume: save(checkpoint,policy,optimizer,config,0,generator,extra)
    for stage_index in range(start,args.stop_after_stage or len(stages)):
        stage = stages[stage_index]
        tick = time.perf_counter()
        row = dict(stage=stage)
        if stage == 'base_pretrain':
            pretrain(policy,optimizer,generator,config['base_steps'],check_time)
            extra['base_state'] = copy.deepcopy(policy.state_dict())
        elif stage == 'cold_sft':
            for _ in range(config['cold_steps']):
                indices = torch.randint(len(train),(16,),generator=generator)
                row['last_loss'] = supervised_step(policy,optimizer,train[indices],target[indices],EOS)
                check_time()
        elif stage in ('reasoning_rl','mixed_rl'):
            reference = frozen_copy(policy)
            for group in optimizer.param_groups: group['lr'] = .0005
            records = []
            rounds = config['rl_steps']*(2 if args.recipe=='zero' else 1)
            for step in range(rounds):
                indices = torch.randint(len(train),(8,),generator=generator)
                use_echo = stage=='mixed_rl' and step%2==1
                batch = echo[indices] if use_echo else train[indices]
                correct_answers = echo[indices,1] if use_echo else truth[indices]
                records.append(reinforcement_step(policy,reference,optimizer,batch,
                    lambda r,m:rewards(r,m,correct_answers),generator,EOS,PAD))
                check_time()
            row['mean_reward'] = sum(r['reward'] for r in records)/rounds
            row['last_kl'] = records[-1]['kl']
            row['generated_completions'] = rounds*8*8
        elif stage == 'rejection':
            teacher = frozen_copy(policy)
            response,mask,_ = sample_completions(teacher,train,8,3,EOS,PAD,generator)
            p,r,lengths = rejection_data(train,response,mask,rewards(response,mask,truth).bool())
            extra.update(rejected_prompts=p,rejected_responses=r,rejected_lengths=lengths)
            row.update(candidates=len(train)*8,unique_verified_responses=len(p))
        elif stage == 'reset_mixed_sft':
            # The report fine-tunes the base checkpoint on the collected mixture.
            policy.load_state_dict(extra['base_state'])
            optimizer = torch.optim.AdamW(policy.parameters(),lr=.003)
            p = torch.cat((extra['rejected_prompts'],echo))
            r = torch.cat((extra['rejected_responses'],echo_target))
            row['verified_reasoning_rows'] = len(extra['rejected_prompts'])
            for _ in range(config['reset_sft_steps']):
                indices = torch.randint(len(p),(16,),generator=generator)
                row['last_loss'] = supervised_step(policy,optimizer,p[indices],r[indices],EOS)
                check_time()
        else:
            family = stage.removeprefix('student_')
            student = r1.build_student(family,vocab=24,width=16,layers=1)
            student_optimizer = torch.optim.AdamW(student.parameters(),lr=.003)
            pretrain(student,student_optimizer,generator,config['student_base_steps'],check_time)
            initial_student = frozen_copy(student)
            p = torch.cat((extra['rejected_prompts'],echo))
            r = torch.cat((extra['rejected_responses'],echo_target))
            for _ in range(config['student_steps']):
                indices = torch.randint(len(p),(16,),generator=generator)
                row['last_loss'] = supervised_step(student,student_optimizer,p[indices],r[indices],EOS)
                check_time()
            row['heldout'] = evaluate(student.eval(),initial_student,prompts[heldout],answers[heldout],args.seed+8000)
            row['verified_reasoning_rows'] = len(extra['rejected_prompts'])
            student_path = checkpoint.replace('checkpoint.pt',f'{family}.pt')
            save(student_path,student,student_optimizer,dict(config,family=family),config['student_steps'],generator)
            extra['students'][family] = student_path
        if not stage.startswith('student_'):
            reference = frozen_copy(policy)
            row['heldout'] = evaluate(policy.eval(),reference,prompts[heldout],answers[heldout],args.seed+8000)
            row['train_greedy'] = evaluate(policy,reference,train,truth,args.seed+8001)['greedy_exact_correctness']
        row['seconds'] = time.perf_counter()-tick
        extra['rows'].append(row)
        save(checkpoint,policy,optimizer,config,stage_index+1,generator,extra)
        print(stage,row.get('heldout',{}).get('greedy_exact_correctness'),flush=True)
    if len(extra['rows']) < len(stages):
        print('Partial stage checkpoint:',checkpoint); return
    result = metadata(config)
    result.update(stages=extra['rows'],checkpoint=checkpoint,students=extra['students'],
                  task='Addition answer token + EOS; separate echo task; no reasoning traces',
                  split='80 training and 20 held-out operand pairs; only training prompts enter rejection data',
                  limitations=['Generated sequence pretraining replaces a pretrained text base',
                               'Scripted answer-only cold start replaces curated reasoning traces',
                               'Mixed RL uses correctness rewards, not human preference reward models',
                               'Students learn verified local teacher answers; no released teacher weights',
                               'No disclosed R1-0528-specific training algorithm is reconstructed'])
    write_json(args.output or f'results/reasoning-{args.recipe}.json',result)


if __name__ == '__main__': main()
