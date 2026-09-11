"""One small wrapper for every mixer: embedding, pre-norm residual blocks, head."""
from torch import nn

from .mamba1 import Mamba1
from .mamba2 import Mamba2
from .mamba3 import Mamba3
from .norm import RMSNorm


class CopyModel(nn.Module):
    def __init__(self, kind, width=32, state_size=8, layers=1, classes=4):
        super().__init__()
        if layers not in (1, 2):
            raise ValueError("use one or two residual blocks")
        self.embedding = nn.Embedding(2 * classes + 2, width)
        self.blocks = nn.ModuleList()
        self.pre_norms = nn.ModuleList()
        for _ in range(layers):
            if kind in ("mamba1", "mamba1_constant"):
                block = Mamba1(width, state_size, selective=kind == "mamba1")
            elif kind == "mamba2":
                block = Mamba2(width, state_size)
            elif kind == "mamba3":
                block = Mamba3(width, state_size)
            else:
                raise ValueError(f"unknown model: {kind}")
            self.blocks.append(block)
            self.pre_norms.append(RMSNorm(width))
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, classes, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for norm, block in zip(self.pre_norms, self.blocks):
            mixed, _ = block(norm(x))
            x = x + mixed
        return self.head(self.norm(x))
