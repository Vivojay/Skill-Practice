"""V4 compressed shared-KV attention, expressed as a readable token loop.

CSA pools a current block and a separately projected previous block, then selects
compressed entries. HCA pools a larger non-overlapping block and uses all entries.
Both include a local window, a zero-valued sink, and output de-rotation.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .rotary import rotate


class Compressor(nn.Module):
    def __init__(self,width,dim,ratio,overlap):
        super().__init__()
        self.ratio, self.overlap = ratio, overlap
        self.current = nn.Linear(width,dim,bias=False)
        self.gate = nn.Linear(width,dim,bias=False)
        self.bias = nn.Parameter(torch.zeros(ratio,dim))
        self.norm = nn.RMSNorm(dim,eps=1e-6)
        if overlap:
            self.previous = nn.Linear(width,dim,bias=False)
            self.previous_gate = nn.Linear(width,dim,bias=False)
            self.previous_bias = nn.Parameter(torch.zeros(ratio,dim))

    def forward(self,tail):
        r = self.ratio
        if tail.shape[1] < r:
            raise ValueError('Only complete blocks may be compressed')
        current = tail[:,-r:]
        values, scores = self.current(current),self.gate(current)+self.bias
        if self.overlap and tail.shape[1] >= 2*r:
            previous = tail[:,-2*r:-r]
            values = torch.cat((self.previous(previous),values),1)
            scores = torch.cat((self.previous_gate(previous)+self.previous_bias,scores),1)
        return self.norm((values*scores.softmax(1)).sum(1,keepdim=True))


@dataclass
class CompressedCache:
    window: torch.Tensor
    entries: torch.Tensor
    index_entries: torch.Tensor
    tail: torch.Tensor
    length: int
    offset: int

    @property
    def nbytes(self):
        tensors = (self.window,self.entries,self.index_entries,self.tail)
        storage = {t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors}
        return sum(storage.values())


class CompressedAttention(nn.Module):
    def __init__(self,width=32,heads=4,dim=8,positional=4,q_rank=16,
                 ratio=2,window=4,top_k=2,overlap=True,groups=2,out_rank=4,rotary=None):
        super().__init__()
        if min(width,heads,dim,positional,q_rank,window,groups,out_rank,top_k) < 1 or ratio < 0:
            raise ValueError('Invalid attention dimensions or budgets')
        if positional > dim or positional % 2 or heads % groups:
            raise ValueError('Need even partial rotary width and divisible head groups')
        self.heads,self.dim,self.positional = heads,dim,positional
        self.ratio,self.window,self.top_k,self.overlap = ratio,window,top_k,overlap
        self.groups,self.out_rank = groups,out_rank
        self.rotary = dict(base=40000. if ratio else 10000.)
        self.rotary.update(rotary or {})
        self.q_down = nn.Linear(width,q_rank,bias=False)
        self.q_norm = nn.RMSNorm(q_rank,eps=1e-6)
        self.q_up = nn.Linear(q_rank,heads*dim,bias=False)
        self.kv = nn.Linear(width,dim,bias=False)
        self.kv_norm = nn.RMSNorm(dim,eps=1e-6)
        self.sink = nn.Parameter(torch.zeros(heads))
        self.output_groups = nn.ModuleList(nn.Linear(heads//groups*dim,out_rank,bias=False) for _ in range(groups))
        self.out = nn.Linear(groups*out_rank,width,bias=False)
        if ratio:
            self.compressor = Compressor(width,dim,ratio,overlap)
        if ratio and overlap:
            self.index_compressor = Compressor(width,dim,ratio,True)
            self.index_query = nn.Linear(q_rank,2*dim,bias=False)
            self.index_weights = nn.Linear(width,2,bias=False)
        self.index_objective = None

    def __getstate__(self):
        state = super().__getstate__()
        state['index_objective'] = None
        return state

    def rotated(self,x,positions):
        r = self.positional
        return torch.cat((x[...,:-r],rotate(x[...,-r:],positions,**self.rotary)),-1)

    def _run(self,x,cache,start):
        b,t,width = x.shape
        if t < 1 or start < 0:
            raise ValueError('Need a nonempty sequence and nonnegative position')
        if cache is None:
            cache = CompressedCache(x.new_empty(b,0,self.dim),x.new_empty(b,0,self.dim),
                                    x.new_empty(b,0,self.dim),x.new_empty(b,0,width),0,start)
        if start != cache.offset+cache.length or cache.window.shape[0] != b or cache.window.dtype != x.dtype or cache.window.device != x.device:
            raise ValueError('Cache must match input and contiguous positions')
        window,entries,index_entries,tail = cache.window,cache.entries,cache.index_entries,cache.tail
        positions = torch.arange(start,start+t,device=x.device)
        cq = self.q_norm(self.q_down(x))
        q = self.q_up(cq).view(b,t,self.heads,self.dim)
        q = q * torch.rsqrt(q.square().mean(-1,keepdim=True)+1e-6)
        q = self.rotated(q,positions)
        kv = self.rotated(self.kv_norm(self.kv(x)).unsqueeze(2),positions).squeeze(2)
        outputs,objectives = [],[]
        for i in range(t):
            length = cache.length+i+1
            window = torch.cat((window,kv[:,i:i+1]),1)[:,-self.window:].clone()
            if self.ratio:
                tail = torch.cat((tail,x[:,i:i+1]),1)
                if length % self.ratio == 0:
                    block_position = positions[i:i+1] + 1-self.ratio
                    entry = self.rotated(self.compressor(tail).unsqueeze(2),block_position).squeeze(2)
                    entries = torch.cat((entries,entry),1)
                    if self.overlap:
                        index_entry = self.rotated(self.index_compressor(tail.detach()).unsqueeze(2),block_position).squeeze(2)
                        index_entries = torch.cat((index_entries,index_entry),1)
                    tail = tail[:,-self.ratio:].clone() if self.overlap else tail[:,:0].clone()
            selected = None
            if self.ratio and self.overlap and entries.shape[1]:
                qi = self.index_query(cq[:,i:i+1].detach()).view(b,1,2,self.dim)
                qi = self.rotated(qi,positions[i:i+1])[:,0]
                weights = self.index_weights(x[:,i].detach())/(2*self.dim)**.5
                index_scores = (torch.einsum('bhd,bsd->bhs',qi,index_entries).relu()*weights[:,:,None]).sum(1)
                selected = index_scores.topk(min(self.top_k,entries.shape[1]),-1).indices
                compressed = entries.gather(1,selected[:,:,None].expand(-1,-1,self.dim))
            else:
                compressed = entries
            keys = torch.cat((window,compressed),1)
            scores = torch.einsum('bhd,bsd->bhs',q[:,i],keys)/self.dim**.5
            probabilities = torch.cat((scores,self.sink[None,:,None].expand(b,-1,-1)),-1).softmax(-1)[...,:-1]
            outputs.append(torch.einsum('bhs,bsd->bhd',probabilities,keys))
            if self.training and selected is not None:
                target = probabilities[:,:,window.shape[1]:].detach().sum(1)
                target = target/target.sum(-1,keepdim=True).clamp_min(1e-30)
                logp = index_scores.gather(1,selected).log_softmax(-1)
                objectives.append((target*(target.clamp_min(1e-30).log()-logp)).sum(-1).mean())
        self.index_objective = torch.stack(objectives).mean() if objectives else x.new_zeros(())
        y = self.rotated(torch.stack(outputs,1),-positions)
        chunks = y.flatten(2).chunk(self.groups,-1)
        y = self.out(torch.cat([projection(chunk) for projection,chunk in zip(self.output_groups,chunks)],-1))
        return y,CompressedCache(window,entries,index_entries,tail,cache.length+t,cache.offset),None

    def forward(self,x,offset=0):
        return self._run(x,None,offset)[0]

    @torch.no_grad()
    def decode(self,x,cache=None,start=None):
        start = (cache.offset+cache.length if cache else 0) if start is None else start
        return self._run(x,cache,start)
