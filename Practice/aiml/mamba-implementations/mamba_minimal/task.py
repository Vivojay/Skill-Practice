"""Locally generated selective copying, with recall-only supervision."""
import torch
from torch.nn import functional as F


def selective_copy(batch_size, length, generator, copies=3, classes=4):
    """IDs: blank=0, relevant=1..K, distractors=K+1..2K, delimiter=2K+1.

    Randomly placed relevant tokens are recalled in their original order.
    Recall inputs are ALL blank: neither labels nor shifted labels enter input.
    Targets are -100 everywhere except after the delimiter.
    """
    prefix = length - copies - 1
    if copies < 1 or prefix < copies or batch_size < 1 or classes < 2:
        raise ValueError("need room for all relevant tokens, delimiter, and recall")
    tokens = torch.zeros(batch_size, length, dtype=torch.long)
    tokens[:, :prefix] = torch.randint(classes+1, 2*classes+1, (batch_size, prefix), generator=generator)
    values = torch.randint(1, classes+1, (batch_size, copies), generator=generator)
    for b in range(batch_size):
        positions = torch.randperm(prefix, generator=generator)[:copies].sort().values
        tokens[b, positions] = values[b]
    tokens[:, prefix] = 2*classes+1
    targets = torch.full_like(tokens, -100)
    targets[:, prefix+1:] = values - 1
    return tokens, targets


def recall_metrics(logits, targets):
    mask = targets != -100
    loss = F.cross_entropy(logits[mask], targets[mask])
    accuracy = (logits[mask].argmax(-1) == targets[mask]).float().mean()
    return loss, accuracy
