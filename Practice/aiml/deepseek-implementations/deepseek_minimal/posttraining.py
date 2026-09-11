"""Small reusable steps for supervised, grouped-RL and rejection-data training."""
import torch
from .grpo import (frozen_copy, completion_logp, sample_completions,
                   group_advantages, grpo_loss, response_mask)


def supervised_step(policy,optimizer,prompts,responses,eos,lengths=None):
    policy.train()
    mask = response_mask(responses,eos,lengths)
    if not mask.any(-1).all():
        raise ValueError('Every response must contain at least one target')
    optimizer.zero_grad()
    logp = completion_logp(policy,prompts,responses)
    loss = -(logp.masked_fill(~mask,0.).sum(-1)/mask.sum(-1)).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
    optimizer.step()
    policy.finish_step()
    return loss.item()


def reinforcement_step(policy,reference,optimizer,prompts,reward_fn,generator,
                       eos,pad,group=8,max_tokens=3,updates=2):
    old = frozen_copy(policy)
    responses,mask,old_logp = sample_completions(old,prompts,group,max_tokens,eos,pad,generator)
    rewards = reward_fn(responses,mask)
    advantages = group_advantages(rewards)
    expanded = prompts.repeat_interleave(group,0)
    with torch.no_grad():
        ref_logp = completion_logp(reference,expanded,responses.flatten(0,1)).reshape_as(old_logp)
    policy.train()
    for _ in range(updates):
        optimizer.zero_grad()
        current = completion_logp(policy,expanded,responses.flatten(0,1)).reshape_as(old_logp)
        loss,info = grpo_loss(current,old_logp,ref_logp,advantages,mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
        optimizer.step()
        policy.finish_step()
    return dict(loss=loss.item(),reward=rewards.mean().item(),kl=info['kl'].item(),
                length=mask.sum(-1).float().mean().item(),
                zero_advantage_groups=advantages.eq(0).all(-1).sum().item())


def rejection_data(prompts,responses,mask,correct):
    """Keep verified complete responses, deduplicating exact prompt/response pairs."""
    selected_prompts,selected_responses,lengths,seen = [],[],[],set()
    for question in range(prompts.shape[0]):
        for sample in range(responses.shape[1]):
            if not bool(correct[question,sample]):
                continue
            length = int(mask[question,sample].sum())
            answer = responses[question,sample,:length]
            key = (tuple(prompts[question].tolist()),tuple(answer.tolist()))
            if key in seen:
                continue
            seen.add(key)
            selected_prompts.append(prompts[question])
            selected_responses.append(responses[question,sample])
            lengths.append(length)
    if not lengths:
        return prompts[:0],responses.reshape(-1,responses.shape[-1])[:0],prompts.new_empty(0)
    return torch.stack(selected_prompts),torch.stack(selected_responses),prompts.new_tensor(lengths)
