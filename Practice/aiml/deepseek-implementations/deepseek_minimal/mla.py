"""V2 MLA equations 9–19 with ordinary adjacent-pair RoPE (no YaRN).

Absorption algebra checked against the official V3 inference implementation;
see THIRD_PARTY.md. Inference caches are explicit values owned by the caller.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .rotary import rotate, attention_scale


def rope(x, positions):
    """x [B,T,H,R], positions [T]; R must be even."""
    return rotate(x, positions)


@dataclass
class Cache:
    latent: torch.Tensor  # [B,S,C] or expanded K [B,S,H,D+R]
    positional: torch.Tensor  # [B,S,R] or expanded V [B,S,H,V]
    offset: int
    expanded: bool = False

    @property
    def length(self):
        return self.latent.shape[1]

    @property
    def nbytes(self):
        # Count backing allocations, not just logical tensor views.
        storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                    for t in (self.latent, self.positional)}
        return sum(storages.values())


class MLA(nn.Module):
    def __init__(self, width=32, heads=4, content=8, positional=4, value=8,
                 kv_rank=8, q_rank=16, rotary=None):
        super().__init__()
        if min(width, heads, content, positional, value, kv_rank, q_rank) < 1 or positional % 2:
            raise ValueError('Positive dimensions and even RoPE dimension required')
        self.heads, self.content, self.positional, self.value = heads, content, positional, value
        self.rotary = dict(rotary or {})
        self.scale = attention_scale(self.rotary.get('factor', 1)) / math.sqrt(content + positional)
        self.q_down = nn.Linear(width, q_rank, bias=False)
        self.q_norm = nn.RMSNorm(q_rank, eps=1e-6)
        self.q_content = nn.Linear(q_rank, heads*content, bias=False)
        self.q_rope = nn.Linear(q_rank, heads*positional, bias=False)
        self.kv_down = nn.Linear(width, kv_rank, bias=False)
        self.kv_norm = nn.RMSNorm(kv_rank, eps=1e-6)
        self.k_up = nn.Linear(kv_rank, heads*content, bias=False)
        self.v_up = nn.Linear(kv_rank, heads*value, bias=False)
        self.k_rope = nn.Linear(width, positional, bias=False)
        self.out = nn.Linear(heads*value, width, bias=False)

    def _project(self, x, start):
        b, t, _ = x.shape
        positions = torch.arange(start, start+t, device=x.device)
        cq = self.q_norm(self.q_down(x))
        qc = self.q_content(cq).view(b,t,self.heads,self.content)
        qr = rotate(self.q_rope(cq).view(b,t,self.heads,self.positional), positions, **self.rotary)
        c = self.kv_norm(self.kv_down(x))
        kr = rotate(self.k_rope(x).unsqueeze(2), positions, **self.rotary).squeeze(2)
        return qc, qr, c, kr

    def _attention(self, x, cache, start, expanded, selected=None):
        if x.ndim != 3 or not x.shape[1] or start < 0:
            raise ValueError('Need nonempty [B,T,D] and nonnegative position')
        if cache is not None:
            if cache.expanded != expanded or start != cache.offset + cache.length:
                raise ValueError('Cache format mismatch or noncontiguous positions')
            if cache.latent.shape[0] != x.shape[0] or cache.latent.dtype != x.dtype or cache.latent.device != x.device:
                raise ValueError('Cache batch, dtype, and device must match input')
        qc, qr, c, kr = self._project(x, start)
        b, t, _ = x.shape
        offset = start if cache is None else cache.offset
        if expanded:
            k = torch.cat((self.k_up(c).view(b,t,self.heads,self.content),
                           kr[:, :, None].expand(-1,-1,self.heads,-1)), -1)
            v = self.v_up(c).view(b,t,self.heads,self.value)
            if cache is not None:
                k, v = torch.cat((cache.latent,k),1), torch.cat((cache.positional,v),1)
            q = torch.cat((qc,qr),-1)
            scores = torch.einsum('bthd,bshd->bhts',q,k)
            new_cache = Cache(k, v, offset, True)
        else:
            # q^T W_UK c; W_UK is position independent.
            absorbed_q = torch.einsum('bthd,hdc->bthc', qc, self.k_up.weight.view(self.heads,self.content,-1))
            if cache is not None:
                c, kr = torch.cat((cache.latent,c),1), torch.cat((cache.positional,kr),1)
            scores = torch.einsum('bthc,bsc->bhts',absorbed_q,c) + torch.einsum('bthr,bsr->bhts',qr,kr)
            new_cache = Cache(c, kr, offset)
        query_pos = torch.arange(start,start+t,device=x.device)
        key_pos = torch.arange(offset,offset+new_cache.length,device=x.device)
        scores = (scores * self.scale).masked_fill(key_pos[None,:] > query_pos[:,None], -torch.inf)
        if selected is not None:
            scores = scores.masked_fill(~selected[:, None], -torch.inf)
        probabilities = scores.softmax(-1)
        if expanded:
            y = self.out(torch.einsum('bhts,bshv->bthv',probabilities,v).flatten(2))
        else:
            context = torch.einsum('bhts,bsc->bthc',probabilities,c)
            # Compose W_O with W_UV without expanding values for prefix tokens.
            absorbed_out = torch.einsum('dhv,hvc->dhc', self.out.weight.view(-1,self.heads,self.value),
                                        self.v_up.weight.view(self.heads,self.value,-1))
            y = torch.einsum('bthc,dhc->btd',context,absorbed_out)
        return y, new_cache, scores

    def forward(self, x, offset=0):
        return self._attention(x, None, offset, True)[0]

    @torch.no_grad()
    def decode(self, x, cache=None, start=None, expanded=False):
        start = (cache.offset+cache.length if cache is not None else 0) if start is None else start
        return self._attention(x, cache, start, expanded)
