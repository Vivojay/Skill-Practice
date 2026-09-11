"""DeepSeekMoE equations 9–17; loss-free balancing Algorithm 1 (v1).

No capacity dropping, device communication, or release-specific group routing.
The residual addition belongs to the caller.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def balance_losses(scores, indices, device_groups=1):
    """Unscaled expert/device losses for one flattened training batch [T,N]."""
    tokens, experts = scores.shape
    if not tokens or experts % device_groups:
        raise ValueError('Need tokens and equal nonempty device groups')
    counts = torch.bincount(indices.flatten(), minlength=experts)
    f = counts.to(scores) * experts / indices.numel()
    p = scores.mean(0)
    expert = (f * p).sum()
    device = (f.view(device_groups, -1).mean(-1)
              * p.view(device_groups, -1).sum(-1)).sum()
    return expert, device


class MoE(nn.Module):
    def __init__(self, width=32, hidden=16, routed=7, shared=1, top_k=3,
                 score='softmax', normalize=False, bias_rate=0., ema_decay=0.,
                 groups=1, top_groups=1, group_score='max', route_scale=1.):
        super().__init__()
        if not 1 <= top_k <= routed or shared < 0:
            raise ValueError('Invalid expert counts')
        if score not in ('softmax', 'sigmoid', 'sqrtsoftplus') or not 0 <= ema_decay < 1 or bias_rate < 0:
            raise ValueError('Invalid router/controller configuration')
        if groups < 1 or routed % groups or not 1 <= top_groups <= groups:
            raise ValueError('Need equal expert groups and valid group selection')
        if top_k > top_groups * (routed // groups) or group_score not in ('max', 'sum2') or route_scale <= 0:
            raise ValueError('Selected groups must contain enough experts')
        self.groups, self.top_groups, self.group_score = groups, top_groups, group_score
        self.route_scale = route_scale
        self.router = nn.Linear(width, routed, bias=False)
        self.experts = nn.ModuleList(Expert(width, hidden) for _ in range(routed))
        self.shared = nn.ModuleList(Expert(width, hidden) for _ in range(shared))
        self.top_k, self.score, self.normalize = top_k, score, normalize
        self.bias_rate, self.ema_decay = bias_rate, ema_decay
        self.register_buffer('selection_bias', torch.zeros(routed))
        self.register_buffer('pending_load', torch.zeros(routed))
        self.register_buffer('ema_load', torch.zeros(routed))
        self.register_buffer('updates', torch.tensor(0))

    def route(self, x):
        logits = self.router(x)
        if self.score == 'softmax':
            scores = logits.softmax(-1)
        elif self.score == 'sigmoid':
            scores = logits.sigmoid()
        else:
            scores = F.softplus(logits).sqrt()
        choice = scores + self.selection_bias
        if self.groups > 1:
            grouped = choice.unflatten(-1, (self.groups, -1))
            count = 1 if self.group_score == 'max' else min(2, grouped.shape[-1])
            group_scores = grouped.topk(count, dim=-1).values.sum(-1)
            keep = torch.zeros_like(group_scores, dtype=torch.bool)
            keep.scatter_(-1, group_scores.topk(self.top_groups, dim=-1).indices, True)
            choice = grouped.masked_fill(~keep[..., None], -torch.inf).flatten(-2)
        indices = choice.topk(self.top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if self.normalize:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return scores, indices, weights * self.route_scale

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        scores, indices, weights = self.route(flat)
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == i)
            if rows.numel():
                out = out.index_add(0, rows, expert(flat[rows]) * weights[rows, slots, None])
        for expert in self.shared:
            out = out + expert(flat)
        loads = torch.bincount(indices.flatten(), minlength=len(self.experts))
        if self.training and self.bias_rate:
            with torch.no_grad():
                self.pending_load.add_(loads)
        return out.reshape_as(x), dict(scores=scores, indices=indices, weights=weights, loads=loads)

    @torch.no_grad()
    def finish_step(self):
        """Call once AFTER optimizer.step, combining all accumulation microbatches.

        EMA decay=0 is the paper's instantaneous update. EMA>0 is our ablation.
        No forward pass changes bias; evaluation changes neither counts nor bias.
        """
        if not self.training or not self.bias_rate or not self.pending_load.sum():
            return
        load = self.pending_load / self.pending_load.sum()
        if self.updates == 0 or self.ema_decay == 0:
            self.ema_load.copy_(load)
        else:
            self.ema_load.lerp_(load, 1 - self.ema_decay)
        # Raw integer counts preserve exact ties in the original sign controller.
        feedback = self.pending_load if self.ema_decay == 0 else self.ema_load
        self.selection_bias.add_(self.bias_rate * (feedback.mean() - feedback).sign())
        self.pending_load.zero_()
        self.updates.add_(1)

    def parameter_counts(self):
        expert_size = sum(p.numel() for p in self.experts[0].parameters())
        return dict(total=sum(p.numel() for p in self.parameters()),
                    active=self.router.weight.numel() + (self.top_k + len(self.shared)) * expert_size,
                    expert_total=(len(self.experts) + len(self.shared)) * expert_size,
                    expert_active=(self.top_k + len(self.shared)) * expert_size)
