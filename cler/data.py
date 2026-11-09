"""Batched histories, requests and time-group pooling."""
import numpy as np
import torch
from torch import nn

def prepare_history_batch(rows, device):
    if not rows:
        raise ValueError('A request batch must not be empty')
    width = max((len(row['history']) for row in rows))
    ids = np.zeros((len(rows), width), dtype=np.int64)
    groups = np.zeros_like(ids)
    for (index, row) in enumerate(rows):
        if len(row['history']) != len(row['groups']):
            raise ValueError('History IDs and groups must have equal lengths')
        ids[index, :len(row['history'])] = row['history']
        groups[index, :len(row['groups'])] = row['groups']
    return (torch.as_tensor(ids, device=device), torch.as_tensor(groups, device=device))

def prepare_request_batch(rows, queries, device):
    (ids, groups) = prepare_history_batch(rows, device)
    query = torch.as_tensor(np.array(queries[[row['query_row'] for row in rows]]), device=device, dtype=torch.float32)
    return (ids, groups, query)

def validate_query(query, batch, text_dim):
    if query.shape != (batch, text_dim) or not torch.isfinite(query).all():
        raise ValueError('Expected finite request vectors [batch,text_dim]')

@torch.no_grad()
def prepare_inputs(backbone, data, queries, projection, batch_size, log):
    from .model import HistoryOperations
    from .localization import project_requests
    proxy = type('HistoryAdapter', (), {'backbone': backbone})()
    records = {}
    device = backbone.item_embeddings.device
    for (split, rows) in data.rows.items():
        vectors = []
        masks = []
        states = []
        texts = []
        directions = []
        width = backbone.model.position_embedding.num_embeddings
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            (ids, groups) = prepare_history_batch(batch, device)
            (v, m, h, active) = HistoryOperations.prepare_history(proxy, ids, groups)
            if not bool(active.all()):
                raise ValueError('Empty history after packing')
            q = torch.tensor(np.array(queries[[r['query_row'] for r in batch]], dtype=np.float32), device=device)
            direction = project_requests(q, projection)
            vectors.append(torch.nn.functional.pad(v, (0, 0, 0, width - v.shape[1])).cpu())
            masks.append(torch.nn.functional.pad(m, (0, width - m.shape[1])).cpu())
            states.append(h.cpu())
            texts.append(q.cpu())
            directions.append(direction.cpu())
        records[split] = {k: torch.cat(v) for (k, v) in zip(('vectors', 'mask', 'h0', 'query', 'request32'), (vectors, masks, states, texts, directions))}
        log.info('Prepared original histories: %s %d', split, len(rows))
    return records

def encode_history(backbone, vectors, mask, group_ids=None):
    active = mask.any(dim=1)
    user = vectors.new_zeros(vectors.size(0), vectors.size(-1))
    if not active.any():
        return (user, active)
    rows = active.nonzero(as_tuple=False).flatten()
    sequences = []
    for b in rows:
        values = vectors[b][mask[b]]
        if group_ids is not None:
            groups = group_ids[b][mask[b]]
            if (groups < 0).any() or (groups[1:] < groups[:-1]).any():
                raise ValueError('history groups must be nonnegative and ordered')
            values = torch.stack([values[groups.eq(g)].mean(dim=0) for g in torch.unique_consecutive(groups)])
        sequences.append(values)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(s) for s in sequences], device=mask.device)
    compact_mask = torch.arange(padded.size(1), device=mask.device)[None, :] < lengths[:, None]
    encoded = backbone.encode(padded, compact_mask)
    return (user.index_copy(0, rows, encoded), active)
