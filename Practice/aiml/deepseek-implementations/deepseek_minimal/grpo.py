"""DeepSeekMath arXiv:2402.03300v1, Eq. 3/4, outcome supervision.

One outer iteration with frozen SFT reference; each fresh group has its own old
policy snapshot. Population reward std is an explicit convention (paper does not
specify correction). No length normalization or KL variant from later trainers.
"""
import copy
import torch


def frozen_copy(policy):
    return copy.deepcopy(policy).eval().requires_grad_(False)


def group_advantages(rewards):
    rewards = rewards.detach()
    std = rewards.std(-1, keepdim=True, correction=0)
    centered = rewards - rewards.mean(-1,keepdim=True)
    return centered / torch.where(std > 0, std, torch.ones_like(std))


def response_mask(tokens, eos, lengths=None):
    """Include first EOS, exclude all later tokens and explicit right padding.

    A PAD sampled *before* EOS is an actual (incorrect) action, not padding.
    For externally padded no-EOS responses, lengths is required.
    """
    end = tokens.eq(eos)
    mask = (end.cumsum(-1)-end.long()).eq(0)
    if lengths is not None:
        mask = mask & (torch.arange(tokens.shape[-1],device=tokens.device) < lengths[...,None])
    return mask


def grpo_loss(logp, old_logp, reference_logp, advantages, mask, clip=.2, beta=.04):
    """Inputs [questions, group, response_length]; advantage [questions,group]."""
    if logp.shape != mask.shape or not mask.any(-1).all():
        raise ValueError('Each response must contain at least one scored action')
    current = logp.masked_fill(~mask,0.)
    old = old_logp.detach().masked_fill(~mask,0.)
    ref = reference_logp.detach().masked_fill(~mask,0.)
    ratio = (current-old).exp()
    adv = advantages.detach()[...,None]
    surrogate = torch.minimum(ratio*adv, ratio.clamp(1-clip,1+clip)*adv)
    delta = ref-current
    kl = delta.exp()-delta-1
    lengths = mask.sum(-1)
    objective = ((surrogate-beta*kl)*mask).sum(-1)/lengths
    return -objective.mean(), dict(kl=((kl*mask).sum(-1)/lengths).mean().detach(),
                                  clip_fraction=(((ratio-1).abs()>clip)*mask).sum().detach()/mask.sum())


def completion_logp(policy, prompts, responses):
    """Flat [B,P], [B,L]; no prompt log probabilities in returned [B,L]."""
    joined = torch.cat((prompts,responses),-1)
    logits = policy(joined[:,:-1])[:,prompts.shape[1]-1:]
    return logits.log_softmax(-1).gather(-1,responses[...,None]).squeeze(-1)


@torch.no_grad()
def sample_completions(policy, prompts, group, max_tokens, eos, pad, generator, greedy=False):
    """Unfiltered, temperature-one on-policy draws; independent samples per prompt."""
    expanded = prompts.repeat_interleave(group,0)
    prefix = expanded
    alive = torch.ones(expanded.shape[0],device=prompts.device,dtype=torch.bool)
    outputs = []
    for _ in range(max_tokens):
        logits = policy(prefix)[:,-1]
        token = logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1),1,generator=generator).squeeze(-1)
        token = torch.where(alive,token,torch.full_like(token,pad))
        outputs.append(token)
        alive = alive & token.ne(eos)
        prefix = torch.cat((prefix,token[:,None]),-1)
    responses = torch.stack(outputs,-1)
    mask = response_mask(responses,eos)
    old_logp = completion_logp(policy,expanded,responses).detach()
    shape = (prompts.shape[0],group,max_tokens)
    return responses.reshape(shape), mask.reshape(shape), old_logp.reshape(shape)
