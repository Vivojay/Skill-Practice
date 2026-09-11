"""One sequential V3 MTP depth, equations 21–25, on a tiny causal backbone."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-6)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.norm2 = nn.RMSNorm(width, eps=1e-6)
        self.ffn = nn.Sequential(nn.Linear(width,2*width), nn.GELU(), nn.Linear(2*width,width))

    def forward(self, x):
        h = self.norm1(x)
        mask = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        x = x + self.attention(h,h,h,attn_mask=mask,need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class TinyLM(nn.Module):
    """Dense single-layer educational backbone, also reused by the GRPO lab."""
    def __init__(self, vocab=16, width=32, heads=4, max_length=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab,width)
        self.position = nn.Embedding(max_length,width)
        self.block = CausalBlock(width,heads)
        self.norm = nn.RMSNorm(width,eps=1e-6)
        self.head = nn.Linear(width,vocab,bias=False)

    def hidden(self, tokens):
        positions = torch.arange(tokens.shape[1],device=tokens.device)
        return self.block(self.embedding(tokens)+self.position(positions))

    def forward(self, tokens):
        return self.head(self.norm(self.hidden(tokens)))


class MTPModel(nn.Module):
    def __init__(self, vocab=16, width=32, heads=4):
        super().__init__()
        self.base = TinyLM(vocab,width,heads)
        self.previous_norm = nn.RMSNorm(width,eps=1e-6)
        self.embedding_norm = nn.RMSNorm(width,eps=1e-6)
        self.combine = nn.Linear(2*width,width,bias=False)
        self.next_block = CausalBlock(width,heads)

    def forward(self, tokens, include_mtp=True):
        h = self.base.hidden(tokens)
        ntp = self.base.head(self.base.norm(h))
        if not include_mtp or tokens.shape[1] < 2:
            return ntp, None
        # h_i^1 sees t_{<=i+1}, and predicts t_{i+2}.
        joined = torch.cat((self.previous_norm(h[:,:-1]),
                            self.embedding_norm(self.base.embedding(tokens[:,1:]))),-1)
        future = self.next_block(self.combine(joined))
        return ntp, self.base.head(self.base.norm(future))


def supervision(tokens, lengths, depth):
    """Targets/masks for depth=0 (NTP) or depth=1 (one additional MTP).

    Contiguous right-padded sequences; lengths includes all real tokens.
    """
    if depth not in (0,1) or (lengths < 1).any() or (lengths > tokens.shape[1]).any():
        raise ValueError('Invalid depth or right-padded sequence lengths')
    target = tokens[:,depth+1:]
    position = torch.arange(depth+1,tokens.shape[1],device=tokens.device)
    return target, position[None,:] < lengths[:,None]


def prediction_losses(ntp, mtp, tokens, lengths):
    """Use the SAME base-prediction denominator at both depths (Eq. 24 1/T).

    For finite right-padded batches T is the total valid NTP prediction count.
    Missing trailing future targets contribute zero, not wraparound targets.
    """
    target, mask = supervision(tokens,lengths,0)
    denominator = mask.sum().clamp_min(1)
    ntp_loss = (F.cross_entropy(ntp[:,:-1].transpose(1,2),target,reduction='none')*mask).sum()/denominator
    if mtp is None or tokens.shape[1] < 3:
        return ntp_loss, ntp_loss*0
    target, mask = supervision(tokens,lengths,1)
    mtp_loss = (F.cross_entropy(mtp[:,:-1].transpose(1,2),target,reduction='none')*mask).sum()/denominator
    return ntp_loss, mtp_loss
