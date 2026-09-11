"""Ordinary pre-normalized decoder blocks and explicit per-layer caches."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from ..moe import MoE
from ..rotary import rotate


@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor
    offset: int

    @property
    def length(self):
        return self.keys.shape[1]

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in (self.keys, self.values))


class GroupedAttention(nn.Module):
    def __init__(self, width=32, heads=4, kv_heads=2, rotary=None,
                 qkv_bias=False, qk_norm=False, norm_eps=1e-6):
        super().__init__()
        if min(width, heads, kv_heads) < 1 or width % heads or heads % kv_heads or (width // heads) % 2:
            raise ValueError('Need even head width and divisible query/KV heads')
        self.heads, self.kv_heads, self.dim = heads, kv_heads, width // heads
        self.rotary = dict(rotary or {})
        self.q = nn.Linear(width, width, bias=qkv_bias)
        self.k = nn.Linear(width, kv_heads*self.dim, bias=qkv_bias)
        self.v = nn.Linear(width, kv_heads*self.dim, bias=qkv_bias)
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(self.dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.dim, eps=norm_eps) if qk_norm else nn.Identity()

    def _attention(self, x, cache=None, start=0):
        b, t, _ = x.shape
        if t < 1 or start < 0:
            raise ValueError('Need nonempty input and nonnegative position')
        if cache is not None and (start != cache.offset + cache.length or
                                  cache.keys.shape[0] != b or cache.keys.device != x.device or
                                  cache.keys.dtype != x.dtype):
            raise ValueError('Cache must be contiguous and match batch, device and dtype')
        positions = torch.arange(start, start+t, device=x.device)
        q = rotate(self.q_norm(self.q(x).view(b,t,self.heads,self.dim)), positions,
                   interleaved=False, **self.rotary)
        k = rotate(self.k_norm(self.k(x).view(b,t,self.kv_heads,self.dim)), positions,
                   interleaved=False, **self.rotary)
        v = self.v(x).view(b,t,self.kv_heads,self.dim)
        offset = start if cache is None else cache.offset
        if cache is not None:
            k, v = torch.cat((cache.keys,k),1), torch.cat((cache.values,v),1)
        new_cache = KVCache(k,v,offset)
        keys = k.repeat_interleave(self.heads // self.kv_heads, dim=2)
        values = v.repeat_interleave(self.heads // self.kv_heads, dim=2)
        scores = torch.einsum('bthd,bshd->bhts',q,keys) / self.dim**.5
        future = torch.arange(offset,offset+k.shape[1],device=x.device)[None] > positions[:,None]
        scores = scores.masked_fill(future, -torch.inf)
        y = torch.einsum('bhts,bshd->bthd',scores.softmax(-1),values)
        return self.out(y.flatten(2)), new_cache, scores

    def forward(self, x, offset=0):
        return self._attention(x, start=offset)[0]

    @torch.no_grad()
    def decode(self, x, cache=None, start=None):
        start = (cache.offset + cache.length if cache else 0) if start is None else start
        return self._attention(x, cache, start)


class Block(nn.Module):
    def __init__(self, width, attention, ffn, norm_eps=1e-6):
        super().__init__()
        self.attention, self.ffn = attention, ffn
        self.attention_norm = nn.RMSNorm(width, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(width, eps=norm_eps)
        self.routing = None

    def __getstate__(self):
        state = super().__getstate__()
        state['routing'] = None
        return state

    def feed_forward(self, x):
        h = self.ffn_norm(x)
        if isinstance(self.ffn, MoE):
            y, self.routing = self.ffn(h)
            self.routing['shape'] = x.shape[:2]
        else:
            y, self.routing = self.ffn(h), None
        return x + y

    def forward(self, x, offset=0, tokens=None):
        return self.feed_forward(x + self.attention(self.attention_norm(x), offset=offset))

    def decode(self, x, cache=None, start=None, tokens=None):
        y, cache, _ = self.attention.decode(self.attention_norm(x), cache, start=start)
        return self.feed_forward(x+y), cache


class Decoder(nn.Module):
    def __init__(self, vocab, width, blocks, norm_eps=1e-6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(width, eps=norm_eps)
        self.head = nn.Linear(width, vocab, bias=False)

    def hidden(self, tokens, offset=0):
        if tokens.ndim != 2 or tokens.shape[1] == 0 or offset < 0:
            raise ValueError('Need nonempty [batch,time] tokens and nonnegative offset')
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, offset, tokens=tokens)
        return x

    def forward(self, tokens, offset=0):
        return self.head(self.norm(self.hidden(tokens, offset)))

    @torch.no_grad()
    def decode(self, tokens, caches=None, start=None):
        if self.training:
            raise RuntimeError('Call eval() before cached decoding')
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError('Need one cache per block')
        if any(c is None for c in caches) and not all(c is None for c in caches):
            raise ValueError('Cannot mix empty and populated layer caches')
        if caches[0] is not None and len({(c.offset,c.length) for c in caches}) != 1:
            raise ValueError('Layer caches must cover the same positions')
        x, updated = self.embed_tokens(tokens), []
        for block, cache in zip(self.blocks, caches):
            x, cache = block.decode(x, cache, start, tokens=tokens)
            updated.append(cache)
        return self.head(self.norm(self.collapse_hidden(x))), updated

    def embed_tokens(self,tokens):
        return self.embedding(tokens)

    def collapse_hidden(self,x):
        return x

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=8):
        if max_new_tokens < 0:
            raise ValueError('Token budget cannot be negative')
        if max_new_tokens == 0:
            return tokens.clone()
        logits, caches = self.decode(tokens)
        result = tokens
        for step in range(max_new_tokens):
            token = logits[:, -1].argmax(-1, keepdim=True)
            result = torch.cat((result, token), 1)
            if step + 1 < max_new_tokens:
                logits, caches = self.decode(token, caches)
        return result

    def finish_step(self):
        for module in self.modules():
            if isinstance(module, MoE):
                module.finish_step()

    def balance_loss(self):
        """Mean sequence balance penalty; caller chooses its coefficient."""
        losses = []
        for block in self.blocks:
            if block.routing is not None:
                b, t = block.routing['shape']
                scores = block.routing['scores'].view(b,t,-1)
                indices = block.routing['indices'].view(b,t,-1)
                scores = scores / scores.sum(-1, keepdim=True)
                counts = F.one_hot(indices, scores.shape[-1]).to(scores).sum(-2).mean(1)
                losses.append(((counts * scores.mean(1)).sum(-1) * scores.shape[-1] / indices.shape[-1]).mean())
        return torch.stack(losses).mean() if losses else self.head.weight.new_zeros(())
