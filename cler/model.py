from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from .localization import block_layout
from .data import validate_query

def resize_banks(model, count):
    if count < 1:
        raise ValueError('Matrix-bank count must be positive')
    transfer = not hasattr(model.backbone, 'item_embeddings')
    sites = [s[2] for s in model.sites] if transfer else [s.module for s in model.backbone.sites]
    current = sites[0].matrices.shape[0]
    if count == current:
        model.num_matrices = count
        return
    old = model.direction
    weights = old.weight.detach().reshape(len(sites), current, model.hidden if hasattr(model, 'hidden') else old.in_features)
    biases = old.bias.detach().reshape(len(sites), current)
    if count <= current:
        old.weight = nn.Parameter(weights[:, :count].reshape(-1, old.in_features).clone())
        old.bias = nn.Parameter(biases[:, :count].reshape(-1).clone())
    else:
        new = nn.Linear(old.in_features, len(sites) * count).to(old.weight)
        nn.init.zeros_(new.weight)
        nn.init.zeros_(new.bias)
        with torch.no_grad():
            new.weight.reshape(len(sites), count, -1)[:, :current].copy_(weights)
            new.bias.reshape(len(sites), count)[:, :current].copy_(biases)
        model.direction = new
    model.direction.out_features = len(sites) * count
    for site in sites:
        previous = site.matrices.detach()
        if count <= current:
            site.matrices = nn.Parameter(previous[:count].clone())
        else:
            values = previous.new_empty(count, *previous.shape[1:])
            nn.init.normal_(values, std=0.02)
            values[:current].copy_(previous)
            site.matrices = nn.Parameter(values)
        site.num_matrices = count
    model.num_matrices = count
    if not transfer:
        model.backbone.num_matrices = count

def _validate_num_matrices(num_matrices: int) -> None:
    if isinstance(num_matrices, bool) or not isinstance(num_matrices, int):
        raise TypeError('num_matrices must be an integer')
    if num_matrices < 1:
        raise ValueError('num_matrices must be positive')

class DirectMatrixLinear(nn.Module):

    def __init__(self, base: nn.Linear, num_matrices: int=4) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('base must be an nn.Linear')
        _validate_num_matrices(num_matrices)
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        base.eval()
        self.num_matrices = num_matrices
        self.matrices = nn.Parameter(base.weight.new_empty(num_matrices, base.out_features, base.in_features))
        nn.init.normal_(self.matrices, std=0.02)
        object.__setattr__(self, 'runtime_coefficients', None)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def set_runtime(self, coefficients: torch.Tensor | None) -> None:
        if coefficients is not None:
            if not isinstance(coefficients, torch.Tensor):
                raise TypeError('runtime coefficients must be a tensor or None')
            if coefficients.ndim != 2 or coefficients.shape[0] < 1 or coefficients.shape[1] != self.num_matrices:
                raise ValueError(f'runtime coefficients must have shape [batch, {self.num_matrices}]')
            if not coefficients.is_floating_point():
                raise TypeError('runtime coefficients must be floating point')
            if coefficients.device != self.matrices.device:
                raise ValueError('runtime coefficients and matrix bank must share a device')
            if coefficients.dtype != self.matrices.dtype:
                raise ValueError('runtime coefficients and matrix bank must share a dtype')
            if not torch.isfinite(coefficients).all().item():
                raise ValueError('runtime coefficients must be finite')
        object.__setattr__(self, 'runtime_coefficients', coefficients)

    def clear_runtime(self) -> None:
        self.set_runtime(None)

    def train(self, mode: bool=True) -> 'DirectMatrixLinear':
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2 or inputs.shape[-1] != self.in_features:
            raise ValueError(f'inputs must have shape [batch, ..., {self.in_features}]')
        coefficients = self.runtime_coefficients
        if coefficients is not None and inputs.shape[0] != coefficients.shape[0]:
            raise ValueError('input and request batch sizes differ')
        base_output = self.base(inputs)
        if coefficients is None:
            return base_output
        minimum_norm = max(1e-08, torch.finfo(self.matrices.dtype).tiny)
        norm = self.matrices.flatten(1).norm(dim=1).clamp_min(minimum_norm)
        normalised = self.matrices / norm[:, None, None]
        mixed = torch.einsum('bm,moi->boi', coefficients, normalised)
        update = torch.einsum('boi,b...i->b...o', mixed, inputs)
        return base_output + update

@dataclass(frozen=True)
class MatrixSite:
    index: int
    layer_index: int
    matrix_name: str
    path: str
    module: DirectMatrixLinear

def install_direct_matrix_sites(model: nn.Module, num_matrices: int=4, matrix_names: Iterable[str]=('W_O', 'W_1', 'W_2')) -> list[MatrixSite]:
    _validate_num_matrices(num_matrices)
    requested = tuple(matrix_names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError('matrix_names must be nonempty and contain no duplicates')
    unknown = set(requested) - {'Q', 'K', 'V', 'W_O', 'W_1', 'W_2'}
    if unknown:
        raise ValueError(f'unknown candidate matrices: {sorted(unknown)}')
    encoder = getattr(model, 'trm_encoder', None)
    layers = getattr(encoder, 'layer', None)
    if layers is None or len(layers) == 0:
        raise TypeError('expected a SASRec with nonempty model.trm_encoder.layer')
    specifications = {'Q': ('multi_head_attention', 'query'), 'K': ('multi_head_attention', 'key'), 'V': ('multi_head_attention', 'value'), 'W_O': ('multi_head_attention', 'dense'), 'W_1': ('feed_forward', 'dense_1'), 'W_2': ('feed_forward', 'dense_2')}
    candidates = []
    for (layer_index, layer) in enumerate(layers):
        for matrix_name in requested:
            (parent_name, attribute) = specifications[matrix_name]
            parent = getattr(layer, parent_name, None)
            original = getattr(parent, attribute, None)
            path = f'trm_encoder.layer.{layer_index}.{parent_name}.{attribute}'
            if not isinstance(original, nn.Linear):
                raise TypeError(f'{path} must be an unwrapped nn.Linear')
            candidates.append((layer_index, matrix_name, path, parent, attribute, original))
    sites = []
    for (layer_index, matrix_name, path, parent, attribute, original) in candidates:
        wrapper = DirectMatrixLinear(original, num_matrices)
        setattr(parent, attribute, wrapper)
        sites.append(MatrixSite(len(sites), layer_index, matrix_name, path, wrapper))
    return sites

class RequestControllerBase(nn.Module):

    def __init__(self, backbone, text_dim=512, hidden=128, scope='ffn_attn'):
        super().__init__()
        if scope not in ('ffn_attn', 'all_six'):
            raise ValueError('Unknown editable site scope')
        (self.backbone, self.text_dim, self.scope) = (backbone, text_dim, scope)
        matrix_names = ('W_1', 'W_2', 'W_O') if scope == 'ffn_attn' else ('Q', 'K', 'V', 'W_O', 'W_1', 'W_2')
        if scope == 'all_six':
            layers = {site.layer_index for site in backbone.sites}
            if not layers or any(({site.matrix_name for site in backbone.sites if site.layer_index == layer} != set(matrix_names) for layer in layers)):
                raise ValueError('all_six requires Q/K/V/WO/W1/W2 editable sites in every layer')
        self.selected_sites = [s.index for s in backbone.sites if s.matrix_name in matrix_names]
        if not self.selected_sites:
            raise ValueError('No FFN or attention output matrix sites selected')
        for site in backbone.sites:
            site.module.matrices.requires_grad_(site.index in self.selected_sites)
        self.request_projection = nn.Linear(text_dim, hidden)
        self.direction = nn.Linear(hidden, len(self.selected_sites) * backbone.num_matrices)
        if scope == 'all_six':
            groups = ('Q', 'K', 'V', 'W_O', 'W_1', 'W_2')
            self.site_strength_indices = [groups.index(backbone.sites[index].matrix_name) for index in self.selected_sites]
            self.strength = nn.Linear(hidden, len(groups))
        else:
            self.site_strength_indices = list(range(len(self.selected_sites)))
            self.strength = nn.Linear(hidden, len(self.selected_sites))
        nn.init.zeros_(self.direction.weight)
        nn.init.zeros_(self.direction.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.model.train(mode)
        return self

class HistoryOperations:

    @torch.no_grad()
    def prepare_history(self, ids, groups=None):
        if ids.ndim != 2 or ids.shape[0] < 1 or ids.dtype != torch.long:
            raise ValueError('ids must be a nonempty batch of int64 sequences')
        if ids.device != self.backbone.item_embeddings.device:
            raise ValueError('ids and backbone must share a device')
        if groups is not None and (groups.shape != ids.shape or groups.device != ids.device or groups.dtype != torch.long):
            raise ValueError('groups must be int64 with the same shape/device as ids')
        if (ids < 0).any() or (ids >= self.backbone.item_embeddings.shape[0]).any():
            raise ValueError('history IDs outside the frozen item table')
        mask = ids.gt(0)
        lengths = mask.sum(-1)
        if not torch.equal(mask, torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]):
            raise ValueError('history IDs must use right padding')
        vectors = F.embedding(ids, self.backbone.item_embeddings)
        sequences = []
        for row in range(len(ids)):
            values = vectors[row, mask[row]]
            if groups is not None and len(values):
                labels = groups[row, mask[row]]
                if (labels < 0).any() or (labels[1:] < labels[:-1]).any():
                    raise ValueError('history groups must be nonnegative and ordered')
                values = torch.stack([values[labels.eq(g)].mean(0) for g in torch.unique_consecutive(labels)])
            sequences.append(values)
        packed = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(row) for row in sequences], device=ids.device)
        packed_mask = torch.arange(packed.shape[1], device=ids.device)[None] < lengths[:, None]
        active = lengths.gt(0)
        h0 = packed.new_zeros(len(ids), self.backbone.hidden_size)
        self.backbone.eval()
        with self.backbone.edit(None):
            if active.any():
                h0[active] = self.backbone.encode(packed[active], packed_mask[active])
        return (packed, packed_mask, h0, active)

    def _relative_norm(self, coefficients, active=None):
        values = []
        if active is None:
            active = torch.ones(len(coefficients), dtype=torch.bool, device=coefficients.device)
        with torch.no_grad():
            for site in self.backbone.sites:
                matrices = site.module.matrices.detach()
                tiny = max(1e-08, torch.finfo(matrices.dtype).tiny)
                support = getattr(site.module, 'runtime_support', None)
                if support is None:
                    bank = matrices.flatten(1)
                    bank = bank / bank.norm(dim=1).clamp_min(tiny)[:, None]
                    gram = bank @ bank.T
                    c = coefficients[:, site.index].detach()
                    delta_norm = torch.einsum('bi,ij,bj->b', c, gram, c).clamp_min(0).sqrt()
                else:
                    (row, col) = support
                    bank = matrices[None] * row[:, None, :, None] * col[:, None, None, :]
                    bank = bank / bank.flatten(2).norm(dim=-1).clamp_min(1e-08)[..., None, None]
                    c = coefficients[active, site.index].detach()
                    delta = torch.einsum('bm,bmoi->boi', c, bank)
                    delta_norm = coefficients.new_zeros(len(coefficients))
                    delta_norm[active] = delta.flatten(1).norm(dim=-1)
                values.append(delta_norm / site.module.base.weight.detach().norm().clamp_min(tiny))
        return torch.stack(values, dim=1)

class HistoryRequestController(RequestControllerBase):
    group_names = ('Q', 'K', 'V', 'WO', 'W1', 'W2')

    def __init__(self, backbone, text_dim=512, hidden=128, use_history=True):
        super().__init__(backbone, text_dim=text_dim, hidden=hidden, scope='all_six')
        if backbone.num_matrices != 4:
            raise ValueError('Controller initialization requires four matrices per site')
        (self.hidden, self.use_history) = (hidden, bool(use_history))
        if self.use_history:
            self.history_projection = nn.Linear(backbone.hidden_size, hidden)
            self.fusion = nn.Linear(4 * hidden, hidden)
        self.last_details = None

    def prepare_history(self, ids, groups=None):
        modes = [(module, module.training) for module in self.backbone.modules()]
        try:
            return HistoryOperations.prepare_history(self, ids, groups)
        finally:
            for (module, training) in modes:
                module.training = training

    def controls(self, query, h0):
        validate_query(query, len(query), self.text_dim)
        if h0.shape != (len(query), self.backbone.hidden_size) or not torch.isfinite(h0).all():
            raise ValueError('h0 must be finite [batch,hidden_size]')
        table = self.backbone.item_embeddings
        if query.device != table.device or h0.device != table.device or query.dtype != table.dtype or (h0.dtype != table.dtype):
            raise ValueError('query/h0 must match backbone device/dtype')
        condition = torch.tanh(self.request_projection(query.detach()))
        if self.use_history:
            h = torch.tanh(self.history_projection(h0.detach()))
            q = condition
            condition = torch.tanh(self.fusion(torch.cat((q, h, q * h, q - h), dim=-1)))
        beta = torch.tanh(self.direction(condition)).reshape(len(query), len(self.selected_sites), self.backbone.num_matrices)
        alpha = torch.sigmoid(self.strength(condition))
        return (beta, alpha)

    def _relative_norm(self, coefficients, active=None):
        if not getattr(self, 'collect_increment_norms', False):
            return None
        return HistoryOperations._relative_norm(self, coefficients, active)

    def encode_prepared(self, vectors, mask, query, h0, *, strength_override=None, detach_direction=False, control_pair=None):
        lengths = self.backbone._validate_history(vectors, mask)
        if len(query) != len(vectors) or h0.shape != (len(vectors), self.backbone.hidden_size):
            raise ValueError('Prepared history, query and h0 batch sizes must match')
        (beta, alpha) = self.controls(query, h0) if control_pair is None else control_pair
        if detach_direction:
            beta = beta.detach()
        effective_alpha = alpha
        if strength_override is not None:
            effective_alpha = torch.as_tensor(strength_override, device=alpha.device, dtype=alpha.dtype)
            if effective_alpha.shape == (alpha.shape[-1],):
                effective_alpha = effective_alpha[None].expand(len(vectors), -1)
            if effective_alpha.shape != alpha.shape:
                raise ValueError(f'strength_override must have shape [{alpha.shape[-1]}] or {list(alpha.shape)}')
            if not torch.isfinite(effective_alpha).all() or (effective_alpha < 0).any():
                raise ValueError('strength_override must be finite and nonnegative')
        active = lengths.gt(0)
        requested = active & query.detach().norm(dim=-1).gt(0)
        coefficients = beta * effective_alpha[:, self.site_strength_indices, None]
        coefficients = coefficients * requested[:, None, None]
        self.backbone.model.train(self.training)
        user = vectors.new_zeros(len(vectors), self.backbone.hidden_size)
        with self.backbone.edit(coefficients[active] if active.any() else None):
            if active.any():
                indices = active.nonzero(as_tuple=False).flatten()
                encoded = self.backbone.encode(vectors[active], mask[active])
                user = user.index_copy(0, indices, encoded)
        regularizer = coefficients[requested].square().mean() if requested.any() else coefficients.sum() * 0
        details = dict(alpha=alpha, beta=beta, effective_alpha=effective_alpha, coefficients=coefficients, active=active, relative_norm=self._relative_norm(coefficients, active))
        self.last_details = details
        return (user, active, regularizer, details)

class SupportedLinear(DirectMatrixLinear):

    def forward(self, inputs):
        gates = getattr(self, 'runtime_gates', None)
        if gates is not None:
            (row, col) = gates
            return F.linear(inputs * col[:, None, :], self.base.weight, self.base.bias) * row[:, None, :]
        support = self.runtime_support
        if support is None or self.runtime_coefficients is None:
            return super().forward(inputs)
        if getattr(self, '_fast_v2_skip_update', False):
            zero = self.matrices.sum() * 0 + self.runtime_coefficients.sum() * 0
            return self.base(inputs) + zero
        (row, col) = support
        banks = self.matrices[None] * row[:, None, :, None] * col[:, None, None, :]
        norm = banks.flatten(2).norm(dim=-1).clamp_min(1e-08)
        normalized = banks / norm[..., None, None]
        mixed = torch.bmm(self.runtime_coefficients.unsqueeze(1), normalized.flatten(2)).reshape(len(inputs), self.out_features, self.in_features)
        if getattr(self, 'protection_eta', 0.0):
            from .protection import soft_project
            mixed = soft_project(mixed, self.protection_basis, self.protection_eta)
        delta = torch.bmm(mixed, inputs.transpose(1, 2)).transpose(1, 2)
        return self.base(inputs) + delta

def install_support(backbone):
    for site in backbone.sites:
        site.module.__class__ = SupportedLinear
        object.__setattr__(site.module, 'runtime_support', None)
        object.__setattr__(site.module, 'runtime_gates', None)

class AllPositionEditor(HistoryRequestController):

    def __init__(self, backbone, method, config):
        super().__init__(backbone, config.get('request_dimension', 512), config['training']['hidden'], use_history=True)
        self.strength = nn.Linear(self.hidden, backbone.num_sites)
        self.site_strength_indices = list(range(backbone.num_sites))
        self.config = config
        self.method = method
        install_support(backbone)
        self.blocks = block_layout(backbone, config['perturbation']['weights_per_block'])
        for s in backbone.sites:
            s.module.matrices.requires_grad_(True)
        self.to(backbone.item_embeddings.device)

    def encode_located(self, vectors, mask, query, h0, selected):
        if selected.shape != (len(vectors), len(self.blocks)) or not ((selected == 0) | (selected == 1)).all():
            raise ValueError('Invalid block mask')
        if not (selected.sum(-1) == self.config['perturbation']['selected_blocks']).all():
            raise ValueError('Incorrect block budget')
        controls = self.controls(query, h0)
        widths = tuple((site.module.out_features for site in self.backbone.sites))
        if not hasattr(self, '_fast_support_row_index'):
            maps = [[-1] * width for width in widths]
            for (number, block) in enumerate(self.blocks):
                for row in range(block['start'], block['stop']):
                    if maps[block['site']][row] != -1:
                        raise ValueError('Overlapping row blocks')
                    maps[block['site']][row] = number
            if any((i < 0 for rows in maps for i in rows)):
                raise ValueError('Incomplete row layout')
            self.register_buffer('_fast_support_row_index', torch.tensor([i for rows in maps for i in rows], dtype=torch.long, device=selected.device), persistent=False)
        active = mask.any(-1)
        rows = selected.index_select(1, self._fast_support_row_index)
        full_rows = rows.split(widths, dim=1)
        enabled = torch.stack([row.any(-1) for row in full_rows], dim=1).to(controls[1].dtype)
        active_rows = rows[active].split(widths, dim=1)
        columns = {site.module.in_features: selected.new_ones(int(active.sum()), site.module.in_features) for site in self.backbone.sites}
        skip = torch.stack([row.any() for row in active_rows]).tolist()
        previous = []
        try:
            for (site, row, used) in zip(self.backbone.sites, active_rows, skip):
                module = site.module
                previous.append((module, getattr(module, 'runtime_support', None), getattr(module, '_fast_v2_skip_update', False)))
                object.__setattr__(module, 'runtime_support', (row, columns[module.in_features]))
                object.__setattr__(module, '_fast_v2_skip_update', not used)
            return self.encode_prepared(vectors, mask, query, h0, strength_override=controls[1] * enabled * getattr(self, 'edit_scale', 1.0), control_pair=controls)
        finally:
            for (module, support, skipped) in previous:
                object.__setattr__(module, 'runtime_support', support)
                object.__setattr__(module, '_fast_v2_skip_update', skipped)

    def budget(self):
        n = self.config['perturbation']['selected_blocks'] * self.config['perturbation']['weights_per_block']
        return dict(trainable=sum((p.numel() for p in self.parameters() if p.requires_grad)), supported_weight_elements_per_request=n, active_matrix_bank_elements_per_request=self.backbone.num_matrices * n, stored_matrix_bank_elements=sum((s.module.matrices.numel() for s in self.backbone.sites)), selection=self.method, control_history=True)

class LocalLinear(nn.Module):

    def __init__(self, weight, bias=None):
        super().__init__()
        (self.weight, self.bias) = (weight, bias)
        (self.in_features, self.out_features) = (weight.shape[1], weight.shape[0])
        self.matrices = nn.Parameter(weight.new_empty(4, *weight.shape))
        nn.init.normal_(self.matrices, std=0.02)
        self.coefficients = self.row_mask = self.replacement = None
        self.cached_delta = None
        self.record = False
        (self.inputs, self.outputs) = ([], [])
        self.call_index = 0

    def delta(self):
        if self.cached_delta is not None:
            return self.cached_delta
        banks = self.matrices[None] * self.row_mask[:, None, :, None]
        norms = banks.flatten(2).norm(dim=-1).clamp_min(1e-08)
        result = torch.einsum('bm,bmoi->boi', self.coefficients, banks / norms[..., None, None])
        if getattr(self, 'protection_eta', 0.0):
            from .protection import soft_project
            result = soft_project(result, self.protection_basis, self.protection_eta)
        return result

    def forward(self, value):
        output = F.linear(value, self.weight, self.bias)
        if self.coefficients is not None:
            output = output + torch.einsum('boi,bli->blo', self.delta(), value.reshape(len(value), -1, self.in_features)).reshape_as(output)
        if self.replacement is not None:
            (start, stop, clean) = self.replacement
            changed = output.clone()
            changed[..., start:stop] = clean[self.call_index][..., start:stop]
            output = changed
        self.call_index += 1
        if self.record:
            self.inputs.append(value)
            self.outputs.append(output)
        return output

class LocalGRUCell(nn.Module):

    def __init__(self, original):
        super().__init__()
        self.input = LocalLinear(original.weight_ih, original.bias_ih)
        self.hidden = LocalLinear(original.weight_hh, original.bias_hh)

    def forward(self, value, state):
        (ir, iz, inn) = self.input(value).chunk(3, -1)
        (hr, hz, hn) = self.hidden(state).chunk(3, -1)
        reset = (ir + hr).sigmoid()
        update = (iz + hz).sigmoid()
        candidate = (inn + reset * hn).tanh()
        return candidate + update * (state - candidate)

def mamba_forward(self, hidden_states, inference_params=None):
    from causal_conv1d import causal_conv1d_fn
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    if inference_params is not None:
        raise ValueError('Framework encoder uses complete history prefixes')
    length = hidden_states.shape[1]
    (x, z) = self.in_proj(hidden_states).transpose(1, 2).chunk(2, 1)
    x = causal_conv1d_fn(x=x.contiguous(), weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias, activation=self.activation)
    projected = self.x_proj(x.transpose(1, 2).reshape(-1, self.d_inner))
    (dt, b, c) = projected.split([self.dt_rank, self.d_state, self.d_state], -1)
    dt = F.linear(dt, self.dt_proj.weight).reshape(len(hidden_states), length, self.d_inner).transpose(1, 2)
    b = b.reshape(len(hidden_states), length, self.d_state).transpose(1, 2).contiguous()
    c = c.reshape(len(hidden_states), length, self.d_state).transpose(1, 2).contiguous()
    y = selective_scan_fn(x, dt, -self.A_log.float().exp(), b, c, self.D.float(), z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
    return self.out_proj(y.transpose(1, 2))

class FrameworkEditor(nn.Module):

    def __init__(self, backbone, weights_per_block=512, selected_blocks=10, hidden=128, text_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.original_parameters = tuple(backbone.parameters())
        for parameter in self.original_parameters:
            parameter.requires_grad_(False)
            parameter.grad = None
        self.sites = []
        if backbone.name == 'GRU4Rec':
            for (layer, original) in enumerate(list(backbone.model.G)):
                cell = LocalGRUCell(original)
                backbone.model.G[layer] = cell
                self.sites.extend([(layer, 'input_gates', cell.input), (layer, 'hidden_gates', cell.hidden)])
        else:
            for (layer, block) in enumerate(backbone.model.mamba_layers):
                mamba = block.mamba
                for name in ('in_proj', 'out_proj'):
                    original = getattr(mamba, name)
                    wrapped = LocalLinear(original.weight, original.bias)
                    setattr(mamba, name, wrapped)
                    self.sites.append((layer, name, wrapped))
                mamba.forward = MethodType(mamba_forward, mamba)
                for name in ('w_1', 'w_2'):
                    original = getattr(block.ffn, name)
                    wrapped = LocalLinear(original.weight, original.bias)
                    setattr(block.ffn, name, wrapped)
                    self.sites.append((layer, name, wrapped))
        self.blocks = []
        for (site, (layer, name, linear)) in enumerate(self.sites):
            if weights_per_block % linear.in_features:
                raise ValueError('Block budget must divide input width')
            rows = weights_per_block // linear.in_features
            if linear.out_features % rows:
                raise ValueError('Candidate rows must divide into equal-parameter blocks')
            for start in range(0, linear.out_features, rows):
                self.blocks.append(dict(site=site, layer=layer, matrix=name, start=start, stop=start + rows, weights=weights_per_block))
        self.selected_blocks = selected_blocks
        if not 0 < selected_blocks <= len(self.blocks):
            raise ValueError('Invalid selected-block budget')
        self.request_projection = nn.Linear(text_dim, hidden)
        self.history_projection = nn.Linear(32, hidden)
        self.fusion = nn.Linear(4 * hidden, hidden)
        self.direction = nn.Linear(hidden, len(self.sites) * 4)
        self.strength = nn.Linear(hidden, len(self.sites))
        self.to(backbone.table.device)

    def reset_calls(self):
        for (_, _, site) in self.sites:
            site.call_index = 0
            (site.inputs, site.outputs) = ([], [])

    def base_encode(self, vectors, lengths):
        self.reset_calls()
        return self.backbone.encode(vectors, lengths)

    @contextmanager
    def capture(self):
        for (_, _, site) in self.sites:
            site.record = True
        try:
            yield
        finally:
            for (_, _, site) in self.sites:
                site.record = False

    @contextmanager
    def restore_activation(self, block, clean):
        site = self.sites[block['site']][2]
        site.replacement = (block['start'], block['stop'], clean)
        try:
            yield
        finally:
            site.replacement = None

    def controls(self, query, h0):
        q = self.request_projection(query.detach()).tanh()
        h = self.history_projection(h0.detach()).tanh()
        combined = self.fusion(torch.cat((q, h, q * h, q - h), -1)).tanh() if getattr(self, 'use_history', True) else q
        beta = self.direction(combined).tanh().reshape(len(query), len(self.sites), getattr(self, 'num_matrices', 4))
        alpha = self.strength(combined).sigmoid()
        return (beta, alpha * getattr(self, 'edit_scale', 1.0))

    def row_masks(self, selected):
        cache = getattr(self, '_fast_row_index_cache', None)
        if cache is None:
            indices = []
            for (site_id, (_, _, site)) in enumerate(self.sites):
                row_index = torch.full((site.out_features,), -1, dtype=torch.long)
                for (block_id, block) in enumerate(self.blocks):
                    if block['site'] == site_id:
                        if row_index[block['start']:block['stop']].ge(0).any():
                            raise ValueError('Overlapping parameter row blocks')
                        row_index[block['start']:block['stop']] = block_id
                if row_index.lt(0).any():
                    raise ValueError('Parameter block layout must cover all output rows')
                indices.append(row_index)
            cache = dict(cpu=indices, devices={})
            object.__setattr__(self, '_fast_row_index_cache', cache)
        key = str(selected.device)
        if key not in cache['devices']:
            cache['devices'][key] = [index.to(selected.device) for index in cache['cpu']]
        return [selected.index_select(1, index) for index in cache['devices'][key]]

    @contextmanager
    def edits(self, query, h0, selected):
        if selected.shape != (len(query), len(self.blocks)) or not bool(((selected == 0) | (selected == 1)).all()):
            raise ValueError('Expected a binary per-user parameter-block mask')
        if not bool(selected.sum(-1).eq(self.selected_blocks).all()):
            raise ValueError('Selected block count differs from budget')
        (beta, alpha) = self.controls(query, h0)
        coefficients = []
        try:
            for (j, (row, (_, _, site))) in enumerate(zip(self.row_masks(selected), self.sites)):
                site.row_mask = row
                site.coefficients = beta[:, j] * alpha[:, j, None] * row.any(-1)[:, None]
                site.cached_delta = site.delta()
                coefficients.append(site.coefficients)
            yield torch.stack(coefficients, 1).square().mean()
        finally:
            for (_, _, site) in self.sites:
                site.row_mask = site.coefficients = None
                site.cached_delta = None

    def encode_edited(self, vectors, lengths, query, h0, selected):
        with self.edits(query, h0, selected) as regularizer:
            user = self.base_encode(vectors, lengths)
        return (user, regularizer)

    def editor_state(self):
        return {name: p.detach().cpu().clone() for (name, p) in self.named_parameters() if p.requires_grad}

    def load_editor(self, state):
        parameters = dict(self.named_parameters())
        if set(state) != {name for (name, p) in parameters.items() if p.requires_grad}:
            raise ValueError('Editor checkpoint parameter set differs')
        with torch.no_grad():
            for (name, value) in state.items():
                parameters[name].copy_(value.to(parameters[name]))


def create_editor(backbone, *, request_dim=1024, hidden=128,
                  weights_per_block=512, selected_blocks=10, banks=4,
                  edit_scale=1.0):
    """Build a local editor around a caller-supplied, pretrained backbone adapter."""
    from .backbone import SASRecBackbone, SequentialBackbone
    for name, value in dict(request_dim=request_dim, hidden=hidden,
                            weights_per_block=weights_per_block,
                            selected_blocks=selected_blocks, banks=banks).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if edit_scale < 0:
        raise ValueError('edit_scale must be nonnegative')
    if isinstance(backbone, SASRecBackbone):
        config = dict(request_dimension=request_dim, training=dict(hidden=hidden),
                      perturbation=dict(weights_per_block=weights_per_block,
                                        selected_blocks=selected_blocks))
        model = AllPositionEditor(backbone, 'causal', config)
    elif isinstance(backbone, SequentialBackbone):
        model = FrameworkEditor(backbone, weights_per_block, selected_blocks,
                                hidden, text_dim=request_dim)
    else:
        raise TypeError('Expected SASRecBackbone or SequentialBackbone')
    if selected_blocks > len(model.blocks):
        raise ValueError('Selected-block budget exceeds available blocks')
    resize_banks(model, banks)
    model.edit_scale = edit_scale
    return model
