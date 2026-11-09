"""Trainable semantic projection and training-history alignment."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SemanticProjection(nn.Module):
    """Shared two-layer MLP mapping semantic vectors to recommendation space."""

    def __init__(self, input_dim=1024, output_dim=32, hidden_dim=128, output_norm=1.0):
        super().__init__()
        if any(not isinstance(d, int) or isinstance(d, bool) or d < 1
               for d in (input_dim, output_dim, hidden_dim)):
            raise ValueError('Projection dimensions must be positive integers')
        if not math.isfinite(output_norm) or output_norm <= 0:
            raise ValueError('output_norm must be finite and positive')
        self.input_dim, self.output_dim, self.hidden_dim = input_dim, output_dim, hidden_dim
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, output_dim))
        self.register_buffer('output_norm', torch.tensor(float(output_norm)))
        self.register_buffer('fitted', torch.tensor(False))

    def forward(self, values):
        if values.ndim < 2 or values.shape[-1] != self.input_dim:
            raise ValueError('Semantic vectors have an incompatible input dimension')
        parameter = next(self.parameters())
        values = values.to(device=parameter.device, dtype=parameter.dtype)
        if not torch.isfinite(values).all():
            raise ValueError('Semantic vectors must be finite')
        return F.normalize(self.network(values), dim=-1, eps=1e-8) * self.output_norm

    def freeze(self):
        if not bool(self.fitted):
            raise ValueError('Train or load the projection before freezing it')
        self.eval()
        self.requires_grad_(False)
        return self


def fit_projection(projection, item_semantics, item_embeddings, training_rows, *,
                   epochs=30, batch_size=256, lr=1e-3, seed=0, log=None):
    """Fit only on training-history IDs; never optimize the supplied item table."""
    if not isinstance(projection, SemanticProjection):
        raise TypeError('Expected SemanticProjection')
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('Training epochs, batch size and learning rate must be positive')
    ids = np.asarray(sorted({int(i) for row in training_rows for i in row['history']
                             if int(i) != 0}), dtype=np.int64)
    if not len(ids) or ids[0] < 1:
        raise ValueError('Training histories must contain positive item IDs')
    if (len(item_semantics.shape) != 2 or len(item_embeddings.shape) != 2
            or item_semantics.shape[1] != projection.input_dim
            or item_embeddings.shape[1] != projection.output_dim
            or ids[-1] >= min(len(item_semantics), len(item_embeddings))):
        raise ValueError('Expected ID-aligned semantic and item tables, including padding row 0')
    parameter = next(projection.parameters())
    device, dtype = parameter.device, parameter.dtype

    def take(table, indices):
        if torch.is_tensor(table):
            return table.index_select(0, torch.as_tensor(indices, device=table.device)).detach().to(device=device, dtype=dtype)
        return torch.as_tensor(np.array(table[indices], copy=True), device=device, dtype=dtype)

    norm_sum = 0.0
    for offset in range(0, len(ids), batch_size):
        target = take(item_embeddings, ids[offset:offset + batch_size])
        if not torch.isfinite(target).all() or bool((target.norm(dim=-1) < 1e-8).any()):
            raise ValueError('Training item embeddings must be finite and nonzero')
        norm_sum += target.norm(dim=-1).sum().item()
    with torch.no_grad():
        projection.output_norm.fill_(norm_sum / len(ids))
        projection.fitted.fill_(False)
    projection.requires_grad_(True).train()
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        total = 0.0
        order = rng.permutation(ids)
        for offset in range(0, len(ids), batch_size):
            selected = order[offset:offset + batch_size]
            target = F.normalize(take(item_embeddings, selected), dim=-1)
            prediction = F.normalize(projection(take(item_semantics, selected)), dim=-1)
            loss = (1 - (prediction * target).sum(-1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite projection loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in projection.parameters()):
                raise FloatingPointError('Nonfinite projection gradients')
            optimizer.step()
            total += loss.item() * len(selected)
        history.append(dict(epoch=epoch + 1, alignment_loss=total / len(ids)))
        if log is not None:
            log.info('Projection epoch=%d alignment_loss=%.6f', epoch + 1, total / len(ids))
    projection.fitted.fill_(True)
    projection.freeze()
    return dict(training_items=len(ids), training_item_ids=ids.tolist(), history=history,
                objective='mean_one_minus_cosine_to_frozen_training_item_embeddings')


def save_projection(projection, path):
    from .utils import checkpoint
    if not bool(projection.fitted):
        raise ValueError('Cannot save an unfitted projection as a trained encoder')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint(path, dict(kind='semantic_mlp_v1',
        architecture=dict(input_dim=projection.input_dim, output_dim=projection.output_dim,
                          hidden_dim=projection.hidden_dim),
        state={k: v.detach().cpu() for k, v in projection.state_dict().items()}))


def load_projection(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'semantic_mlp_v1':
        raise ValueError('Expected a trained semantic MLP checkpoint')
    projection = SemanticProjection(**saved['architecture']).to(device)
    projection.load_state_dict(saved['state'], strict=True)
    return projection.freeze()
