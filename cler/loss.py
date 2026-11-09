"""Click-pair sampling and click-ranking and coefficient objectives."""
import numpy as np
import torch
from torch.nn import functional as F


def click_pairs(rows, catalog, rng, device):
    owners, positives, negatives = [], [], []
    for owner, row in enumerate(rows):
        banned = set(map(int,row['history'])) | set(map(int,row['positive']))
        if len(banned) >= catalog:
            raise ValueError('No eligible negative item')
        for positive in row['positive']:
            negative = int(rng.integers(1,catalog+1))
            while negative in banned:
                negative = int(rng.integers(1,catalog+1))
            owners.append(owner)
            positives.append(int(positive))
            negatives.append(negative)
    return tuple(torch.tensor(v, dtype=torch.long, device=device) for v in (owners,positives,negatives))


def click_loss(user, table, owners, positives, negatives, score_function=None):
    if score_function is None:
        difference = (user[owners] * (table[positives]-table[negatives])).sum(-1)
        return F.softplus(-difference).mean()
    scores = score_function(user[owners], torch.stack((positives,negatives),-1))
    return F.softplus(scores[:,1]-scores[:,0]).mean()


def editor_loss(click, regularizer, weight):
    return click + weight * regularizer


def finite_gradients(parameters):
    flags = [torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]
    return bool(torch.stack(flags).all()) if flags else True
