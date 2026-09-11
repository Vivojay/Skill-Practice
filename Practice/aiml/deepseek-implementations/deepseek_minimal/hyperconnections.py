"""Expanded residual streams with finite Sinkhorn normalization (V4, eqs. 1-8)."""
import torch
from torch import nn


def sinkhorn(logits, iterations=20):
    if iterations < 1:
        raise ValueError('Need at least one normalization iteration')
    # Log space avoids overflow without changing the alternating normalization.
    z = logits
    for _ in range(iterations):
        z = z - z.logsumexp(-2,keepdim=True)
        z = z - z.logsumexp(-1,keepdim=True)
    return z.exp()


class HyperConnection(nn.Module):
    def __init__(self, width, streams=2, iterations=20):
        super().__init__()
        if streams < 2:
            raise ValueError('Use an ordinary residual for one stream')
        self.streams, self.iterations = streams, iterations
        self.norm = nn.RMSNorm(width*streams,eps=1e-6,elementwise_affine=False)
        self.pre = nn.Linear(width*streams,streams,bias=False)
        self.post = nn.Linear(width*streams,streams,bias=False)
        self.residual = nn.Linear(width*streams,streams*streams,bias=False)
        self.scales = nn.Parameter(torch.full((3,),.01))
        self.pre_bias = nn.Parameter(torch.full((streams,),-float(torch.tensor(streams-1).log())))
        self.post_bias = nn.Parameter(torch.zeros(streams))
        self.residual_bias = nn.Parameter(2*torch.eye(streams))

    def maps(self, x):
        h = self.norm(x.flatten(-2))
        pre = (self.scales[0]*self.pre(h)+self.pre_bias).sigmoid()
        post = 2*(self.scales[1]*self.post(h)+self.post_bias).sigmoid()
        residual = self.scales[2]*self.residual(h).unflatten(-1,(self.streams,self.streams)) + self.residual_bias
        return pre,post,sinkhorn(residual,self.iterations)

    def split(self, x):
        pre,post,residual = self.maps(x)
        return (pre[...,None]*x).sum(-2),post,residual

    def combine(self, x, update, post, residual):
        return torch.einsum('...ij,...jd->...id',residual,x) + post[...,None]*update.unsqueeze(-2)


class Collapse(nn.Module):
    def __init__(self,width,streams):
        super().__init__()
        self.norm = nn.RMSNorm(width*streams,eps=1e-6,elementwise_affine=False)
        self.projection = nn.Linear(width*streams,streams)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self,x):
        weights = self.projection(self.norm(x.flatten(-2))).sigmoid()
        return (weights[...,None]*x).sum(-2)
