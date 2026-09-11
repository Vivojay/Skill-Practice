"""Original dense family and the early MoE family, at a small local scale."""
from .decoder import Block, Decoder, GroupedAttention
from ..moe import Expert, MoE


def build(vocab=32, width=32, layers=2, heads=4, kv_heads=4, moe=False):
    if layers < 1:
        raise ValueError('Need at least one layer')
    blocks = []
    for layer in range(layers):
        attention = GroupedAttention(width, heads, kv_heads)
        ffn = MoE(width, hidden=width//2, routed=8, shared=1, top_k=2) if moe and layer else Expert(width, 3*width)
        blocks.append(Block(width, attention, ffn))
    return Decoder(vocab, width, blocks)
