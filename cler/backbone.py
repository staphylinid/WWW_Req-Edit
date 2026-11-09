"""Adapters for pretrained sequential recommenders supplied by the caller."""
from contextlib import contextmanager
import torch
import numpy as np
from torch import nn
from .model import install_direct_matrix_sites

class SequentialBackbone(nn.Module):
    def __init__(self, name, model):
        super().__init__()
        if name not in ('GRU4Rec', 'Mamba4Rec'):
            raise ValueError('Unknown backbone type')
        self.name, self.model = name, model

    @property
    def table(self):
        return self.model.Wy.weight if self.name == 'GRU4Rec' else self.model.item_embedding.weight
    def encode(self, vectors, lengths):
        if self.name == 'Mamba4Rec':
            hidden = self.model.LayerNorm(self.model.dropout(vectors))
            for layer in self.model.mamba_layers:
                hidden = layer(hidden)
            return self.model.gather_indexes(hidden, lengths - 1)
        states = [vectors.new_zeros(len(vectors), 32) for _ in self.model.layers]
        for t in range(vectors.shape[1]):
            active = (t < lengths)[:, None]
            value = self.model.DE(vectors[:, t])
            for (j, cell) in enumerate(self.model.G):
                value = self.model.D[j](cell(value, states[j]))
                states[j] = torch.where(active, value, states[j])
        return states[-1]
    def scores(self, user, ids):
        values = (user[:, None] * torch.nn.functional.embedding(ids, self.table)).sum(-1)
        if self.name == 'GRU4Rec':
            values = values + self.model.By(ids).squeeze(-1)
        return values

def create_sasrec(item_count, *, hidden_size=32, max_history=30, n_layers=2, n_heads=2, inner_size=None, hidden_dropout=0.2, attention_dropout=0.2, device='cpu'):
    if hidden_size < 2 or n_heads < 1 or hidden_size % n_heads:
        raise ValueError('hidden_size must be positive and divisible by n_heads')
    if item_count < 1 or max_history < 1 or n_layers < 1:
        raise ValueError('Item count, history capacity and layer count must be positive')
    import numpy as np
    for (name, replacement) in (('float_', np.float64), ('complex_', np.complex128), ('unicode_', np.str_)):
        if not hasattr(np, name):
            setattr(np, name, replacement)
    from recbole.model.sequential_recommender.sasrec import SASRec

    class CatalogShape:

        def num(self, field):
            return item_count + 1 if field == 'item_id' else 1
    config = dict(USER_ID_FIELD='user_id', ITEM_ID_FIELD='item_id', NEG_PREFIX='neg_', LIST_SUFFIX='_list', ITEM_LIST_LENGTH_FIELD='item_length', MAX_ITEM_LIST_LENGTH=max_history, device=str(device), n_layers=n_layers, n_heads=n_heads, hidden_size=hidden_size, embedding_size=hidden_size, inner_size=inner_size if inner_size is not None else 4 * hidden_size, hidden_dropout_prob=hidden_dropout, attn_dropout_prob=attention_dropout, hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02, loss_type='BPR')
    return SASRec(config, CatalogShape()).to(device)

def load_sasrec(base_state, *, device='cpu', editable=False, num_matrices=4, n_heads=2, hidden_dropout=0.2, attention_dropout=0.2, editable_scope='ffn_attn'):
    table = base_state['model.item_embedding.weight']
    positions = base_state['model.position_embedding.weight']
    layers = {int(key.split('.')[3]) for key in base_state if key.startswith('model.trm_encoder.layer.')}
    inner = base_state['model.trm_encoder.layer.0.feed_forward.dense_1.weight'].shape[0]
    original = create_sasrec(table.shape[0] - 1, hidden_size=table.shape[1], max_history=positions.shape[0], n_layers=len(layers), n_heads=n_heads, inner_size=inner, hidden_dropout=hidden_dropout, attention_dropout=attention_dropout, device=device)
    original.load_state_dict({key.removeprefix('model.'): value for (key, value) in base_state.items()}, strict=True)
    return SASRecBackbone(original, freeze=True, editable=editable, num_matrices=num_matrices, editable_scope=editable_scope)

class SASRecBackbone(nn.Module):

    def __init__(self, model, *, freeze=True, editable=False, num_matrices=4, editable_scope='ffn_attn'):
        super().__init__()
        if editable and (not freeze):
            raise ValueError('Matrix editing requires a frozen base')
        scopes = {'ffn_attn': ('W_O', 'W_1', 'W_2'), 'all_six': ('Q', 'K', 'V', 'W_O', 'W_1', 'W_2')}
        if editable_scope not in scopes:
            raise ValueError('editable_scope must be ffn_attn or all_six')
        self.model = model
        self.original_parameters = tuple(model.parameters())
        self.num_matrices = num_matrices
        (self.frozen, self.editable) = (freeze, editable)
        self.editable_scope = editable_scope
        self.sites = []
        if freeze:
            for parameter in self.original_parameters:
                parameter.requires_grad_(False)
        if editable:
            self.sites = install_direct_matrix_sites(model, num_matrices, scopes[editable_scope])
        self.train(not freeze)

    @property
    def num_sites(self):
        return len(self.sites)

    @property
    def hidden_size(self):
        return self.model.item_embedding.embedding_dim

    @property
    def item_embeddings(self):
        return self.model.item_embedding.weight[:self.model.n_items]

    def freeze_original(self):
        self.frozen = True
        for parameter in self.original_parameters:
            parameter.requires_grad_(False)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
            for site in self.sites:
                site.module.train(mode)
        return self

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self.model, 'device'):
            self.model.device = next(self.model.parameters()).device
        return result

    def _validate_history(self, vectors, mask):
        if vectors.ndim != 3 or vectors.shape[0] < 1 or vectors.shape[2] != self.hidden_size:
            raise ValueError('history_vectors must be [nonempty batch,length,hidden_size]')
        if mask.dtype != torch.bool or mask.shape != vectors.shape[:2]:
            raise ValueError('history_mask must be boolean [batch,length]')
        table = self.item_embeddings
        if vectors.device != table.device or mask.device != vectors.device or vectors.dtype != table.dtype:
            raise ValueError('history vectors, mask and base model must have compatible device/dtype')
        if not torch.isfinite(vectors).all().item():
            raise ValueError('history vectors must be finite, including padding')
        lengths = mask.sum(-1)
        prefix = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        if not torch.equal(prefix, mask):
            raise ValueError('history must be a contiguous prefix with right padding')
        return lengths

    def encode(self, history_vectors, history_mask):
        lengths = self._validate_history(history_vectors, history_mask)
        if (lengths == 0).any():
            raise ValueError('SASRec encoding requires nonempty history')
        (batch, width, _) = history_vectors.shape
        if width > self.model.position_embedding.num_embeddings:
            raise ValueError('History exceeds position embedding capacity')
        positions = torch.arange(width, device=history_vectors.device)
        embedded = history_vectors + self.model.position_embedding(positions)[None]
        embedded = self.model.dropout(self.model.LayerNorm(embedded))
        causal = torch.ones((width, width), dtype=torch.bool, device=history_vectors.device).tril()
        allowed = history_mask[:, None, None, :] & causal[None, None]
        attention_mask = (~allowed).to(embedded.dtype) * -10000.0
        tokens = self.model.trm_encoder(embedded, attention_mask, output_all_encoded_layers=True)[-1]
        return tokens[torch.arange(batch, device=tokens.device), lengths - 1]

    @contextmanager
    def edit(self, coefficients):
        if coefficients is not None:
            if not self.editable:
                raise ValueError('this base was loaded without editable SASRec sites')
            expected = (self.num_sites, self.num_matrices)
            if coefficients.ndim != 3 or coefficients.shape[0] < 1 or coefficients.shape[1:] != expected:
                raise ValueError(f'coefficients must be [nonempty batch,{expected[0]},{expected[1]}]')
            if coefficients.dtype != self.item_embeddings.dtype or coefficients.device != self.item_embeddings.device:
                raise ValueError('coefficient device/dtype must match the backbone')
            if not torch.isfinite(coefficients).all().item():
                raise ValueError('coefficients must be finite')
            for site in self.sites:
                if site.module.num_matrices != self.num_matrices:
                    raise ValueError('Site matrix bank count differs from coefficient width')
        previous = [site.module.runtime_coefficients for site in self.sites]
        try:
            for site in self.sites:
                object.__setattr__(site.module, 'runtime_coefficients', None if coefficients is None else coefficients[:, site.index])
            yield self
        finally:
            for (site, old) in zip(self.sites, previous):
                object.__setattr__(site.module, 'runtime_coefficients', old)

def history_vectors(rows, table):
    width = max((len(r['history']) for r in rows))
    ids = np.zeros((len(rows), width), dtype=np.int64)
    groups = np.zeros_like(ids)
    lengths = []
    for (j, row) in enumerate(rows):
        n = len(row['history'])
        ids[j, :n] = row['history']
        (unique, inverse) = np.unique(row['groups'], return_inverse=True)
        groups[j, :n] = inverse
        lengths.append(len(unique))
    device = table.device
    ids = torch.as_tensor(ids, device=device)
    groups = torch.as_tensor(groups, device=device)
    valid = ids.ne(0)
    values = torch.nn.functional.embedding(ids, table) * valid[..., None]
    sums = values.new_zeros(values.shape).scatter_add_(1, groups[..., None].expand_as(values), values)
    counts = values.new_zeros(ids.shape).scatter_add_(1, groups, valid.to(values.dtype))
    vectors = sums / counts.clamp_min(1)[..., None]
    return (vectors[:, :max(lengths)], torch.tensor(lengths, device=device))

def pairwise_loss(scores, positive_counts, negative_count):
    counts = torch.as_tensor(positive_counts, device=scores.device)
    p = int(counts.max())
    positive = scores[:, :p]
    negative = scores.gather(1, counts[:, None] + torch.arange(negative_count, device=scores.device)[None])
    weight = (torch.arange(p, device=scores.device)[None] < counts[:, None])[..., None]
    values = torch.nn.functional.softplus(negative[:, None] - positive[:, :, None]) * weight
    return (values.sum((1, 2)) / (counts * negative_count)).mean()
