"""One sequential MTP depth attached to a real family backbone."""
from copy import deepcopy
import torch
from torch import nn


class MultiTokenModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        width = base.embedding.embedding_dim
        self.previous_norm = nn.RMSNorm(width,eps=1e-6)
        self.embedding_norm = nn.RMSNorm(width,eps=1e-6)
        self.combine = nn.Linear(2*width,width,bias=False)
        self.next_block = deepcopy(base.blocks[-1])

    def forward(self, tokens):
        h = self.base.hidden(tokens)
        ntp = self.base.head(self.base.norm(h))
        if tokens.shape[1] < 2:
            return ntp,None
        joined = torch.cat((self.previous_norm(h[:,:-1]),
                            self.embedding_norm(self.base.embedding(tokens[:,1:]))),-1)
        h = self.next_block(self.combine(joined))
        return ntp,self.base.head(self.base.norm(h))

    def finish_step(self):
        from ..moe import MoE
        for module in self.modules():
            if isinstance(module, MoE):
                module.finish_step()
