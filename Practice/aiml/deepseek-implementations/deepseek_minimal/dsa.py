"""V3.2 token selection and its detached attention-distribution supervision.

This readable version evaluates a dense mask. It checks the sparse mathematics,
but does not claim the memory or throughput of a fused gather attention kernel.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .mla import MLA, Cache
from .rotary import rotate


def select_tokens(scores, allowed, top_k):
    if top_k < 1:
        raise ValueError('Need a positive selection budget')
    indices = scores.masked_fill(~allowed, -torch.inf).topk(min(top_k,scores.shape[-1]),-1).indices
    selected = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1,indices,True)
    return selected & allowed


def index_loss(scores, attention_scores, selected):
    """KL(target || indexer), averaged over queries; target is always detached."""
    target = attention_scores.detach().softmax(-1).sum(1)
    target = target / target.sum(-1,keepdim=True)
    logp = scores.masked_fill(~selected,-torch.inf).log_softmax(-1)
    logp = logp.masked_fill(~selected,0.)
    return (target * (target.clamp_min(1e-30).log()-logp)).sum(-1).mean()


class Indexer(nn.Module):
    def __init__(self, width, q_rank, heads=2, dim=8, positional=4, rotary=None):
        super().__init__()
        if heads < 1 or not 0 < positional <= dim or positional % 2:
            raise ValueError('Invalid indexer dimensions')
        self.heads, self.dim, self.positional = heads, dim, positional
        self.rotary = dict(rotary or {})
        self.q = nn.Linear(q_rank,heads*dim,bias=False)
        self.k = nn.Linear(width,dim,bias=False)
        self.k_norm = nn.LayerNorm(dim,eps=1e-6)
        self.weights = nn.Linear(width,heads,bias=False)

    def project(self, x, cq, start):
        x, cq = x.detach(), cq.detach()
        b,t,_ = x.shape
        p = torch.arange(start,start+t,device=x.device)
        q = self.q(cq).view(b,t,self.heads,self.dim)
        k = self.k_norm(self.k(x)).unsqueeze(2)
        r = self.positional
        # The published indexer uses split-half rotary layout, unlike MLA.
        q = torch.cat((rotate(q[...,:r],p,interleaved=False,**self.rotary),q[...,r:]),-1)
        k = torch.cat((rotate(k[...,:r],p,interleaved=False,**self.rotary),k[...,r:]),-1)
        w = self.weights(x) / (self.heads*self.dim)**.5
        return q,k.squeeze(2),w

    def scores(self, q, k, weights):
        return (torch.einsum('bthd,bsd->bths',q,k).relu()*weights[...,None]).sum(2)


@dataclass
class SparseCache:
    attention: Cache
    index_keys: torch.Tensor

    @property
    def offset(self):
        return self.attention.offset

    @property
    def length(self):
        return self.attention.length

    @property
    def nbytes(self):
        return self.attention.nbytes + self.index_keys.numel()*self.index_keys.element_size()


class SparseMLA(MLA):
    def __init__(self, width=32, top_k=4, **kwargs):
        super().__init__(width=width,**kwargs)
        self.indexer = Indexer(width,self.q_down.out_features,positional=self.positional,rotary=self.rotary)
        self.top_k, self.dense_warmup = top_k, False
        self.index_objective = None

    def _sparse(self, x, cache, start):
        q,k,w = self.indexer.project(x,self.q_norm(self.q_down(x)),start)
        if cache is not None:
            k = torch.cat((cache.index_keys,k),1)
        scores = self.indexer.scores(q,k,w)
        offset = cache.offset if cache else start
        allowed = torch.arange(offset,offset+k.shape[1],device=x.device)[None,:] <= torch.arange(start,start+x.shape[1],device=x.device)[:,None]
        allowed = allowed.unsqueeze(0).expand(x.shape[0],-1,-1)
        selected = allowed if self.dense_warmup else select_tokens(scores,allowed,self.top_k)
        y, attention, logits = self._attention(x,cache.attention if cache else None,start,False,selected)
        self.index_objective = index_loss(scores,logits,selected) if self.training else None
        return y,SparseCache(attention,k),logits

    def forward(self, x, offset=0):
        return self._sparse(x,None,offset)[0]

    @torch.no_grad()
    def decode(self, x, cache=None, start=None):
        start = (cache.offset+cache.length if cache else 0) if start is None else start
        return self._sparse(x,cache,start)
