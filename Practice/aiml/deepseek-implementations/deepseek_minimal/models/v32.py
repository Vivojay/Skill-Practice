"""V3.2 and V3.2-Exp share the DSA backbone; their training recipes differ."""
from . import v3
from ..dsa import SparseMLA


def build(vocab=32, width=32, layers=2, top_k=4, rotary=None):
    model = v3.build(vocab,width,layers,rotary)
    for block in model.blocks:
        block.attention = SparseMLA(width,top_k,rotary=rotary)
    return model
