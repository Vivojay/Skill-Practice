"""V4 text architecture at small scale; see notes/models.md for substitutions."""
from copy import deepcopy
import torch
from torch import nn
from .decoder import Decoder
from ..compressed_attention import CompressedAttention
from ..hyperconnections import HyperConnection, Collapse
from ..moe import MoE


class Block(nn.Module):
    def __init__(self,width,vocab,kind,streams,hash_routing=False,rotary=None):
        super().__init__()
        ratio = {'window':0,'csa':2,'hca':8}[kind]
        self.attention = CompressedAttention(width,ratio=ratio,overlap=kind=='csa',rotary=rotary)
        self.attention_connection = HyperConnection(width,streams)
        self.ffn_connection = HyperConnection(width,streams)
        self.attention_norm = nn.RMSNorm(width,eps=1e-6)
        self.ffn_norm = nn.RMSNorm(width,eps=1e-6)
        self.ffn = MoE(width,width//2,routed=8,top_k=2,score='sqrtsoftplus',
                       normalize=True,route_scale=1.5,bias_rate=0. if hash_routing else .001)
        for expert in list(self.ffn.experts)+list(self.ffn.shared):
            expert.limit = 10.
        # Explicit local hash, not the undistributed production token-to-expert table.
        table = (torch.arange(vocab)[:,None]*3+torch.arange(2)[None]) % 8
        self.register_buffer('hash_table',table if hash_routing else None)
        self.routing = None

    def __getstate__(self):
        state = super().__getstate__()
        state['routing'] = None
        return state

    def feed_forward(self,x,tokens):
        h,post,residual = self.ffn_connection.split(x)
        indices = self.hash_table[tokens].reshape(-1,2) if self.hash_table is not None else None
        h,self.routing = self.ffn(self.ffn_norm(h),indices)
        self.routing['shape'] = x.shape[:2]
        return self.ffn_connection.combine(x,h,post,residual)

    def forward(self,x,offset=0,tokens=None):
        h,post,residual = self.attention_connection.split(x)
        h = self.attention(self.attention_norm(h),offset)
        return self.feed_forward(self.attention_connection.combine(x,h,post,residual),tokens)

    def decode(self,x,cache=None,start=None,tokens=None):
        h,post,residual = self.attention_connection.split(x)
        h,cache,_ = self.attention.decode(self.attention_norm(h),cache,start)
        return self.feed_forward(self.attention_connection.combine(x,h,post,residual),tokens),cache


class Model(Decoder):
    def __init__(self,vocab=32,width=32,kinds=('csa','hca'),streams=2,rotary=None):
        if not kinds or any(kind not in ('window','csa','hca') for kind in kinds):
            raise ValueError('Choose nonempty window/CSA/HCA layers')
        super().__init__(vocab,width,[Block(width,vocab,k,streams,i==0,rotary) for i,k in enumerate(kinds)])
        self.streams = streams
        self.collapse = Collapse(width,streams)

    def embed_tokens(self,tokens):
        return self.embedding(tokens).unsqueeze(2).expand(-1,-1,self.streams,-1)

    def collapse_hidden(self,x):
        return self.collapse(x)

    def stream_hidden(self,tokens,offset=0):
        x = self.embed_tokens(tokens)
        for block in self.blocks:
            x = block(x,offset,tokens)
        return x

    def hidden(self,tokens,offset=0):
        return self.collapse(self.stream_hidden(tokens,offset))


class MultiTokenModel(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base = base
        width = base.embedding.embedding_dim
        self.h_norm = nn.RMSNorm(width,eps=1e-6)
        self.e_norm = nn.RMSNorm(width,eps=1e-6)
        self.h_projection = nn.Linear(width,width,bias=False)
        self.e_projection = nn.Linear(width,width,bias=False)
        self.next_block = deepcopy(base.blocks[-1])
        self.collapse = Collapse(width,base.streams)

    def forward(self,tokens):
        h = self.base.stream_hidden(tokens)
        ntp = self.base.head(self.base.norm(self.base.collapse(h)))
        if tokens.shape[1] < 2:
            return ntp,None
        shifted = tokens[:,1:]
        x = self.h_projection(self.h_norm(h[:,:-1]))
        x = x + self.e_projection(self.e_norm(self.base.embedding(shifted))).unsqueeze(2)
        x = self.next_block(x,tokens=shifted)
        return ntp,self.base.head(self.base.norm(self.collapse(x)))

    def finish_step(self):
        for module in self.modules():
            if isinstance(module,MoE):
                module.finish_step()


def build(vocab=32,width=32,layers=2,rotary=None):
    return Model(vocab,width,tuple('csa' if i%2==0 else 'hca' for i in range(layers)),rotary=rotary)
