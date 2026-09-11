"""V2: MLA and unnormalized softmax routing with a group maximum selector."""
from .decoder import Block, Decoder
from ..mla import MLA
from ..moe import Expert, MoE


def build(vocab=32, width=32, layers=2, rotary=None):
    if layers < 1:
        raise ValueError('Need at least one layer')
    blocks = []
    for layer in range(layers):
        attention = MLA(width=width, rotary=rotary, mscale=.707)
        ffn = Expert(width, 3*width) if layer == 0 else MoE(
            width, hidden=width//2, routed=8, shared=2, top_k=2,
            normalize=False, groups=2, top_groups=1, group_score='max', route_scale=4.)
        blocks.append(Block(width, attention, ffn))
    return Decoder(vocab, width, blocks)
