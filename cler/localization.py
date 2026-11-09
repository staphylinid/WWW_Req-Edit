"""Train-user counterfactual restoration and fixed local edit support."""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F

def locate(model, prepared, config, seed, folder, log):
    """Average training-user restoration effects and select a shared block support."""
    from pathlib import Path
    from .utils import write, bind
    import hashlib
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    blocks = model.blocks
    count = config['perturbation']['selected_blocks']
    if not 0 < count <= len(blocks):
        raise ValueError('Selected block budget exceeds available blocks')
    transfer = not hasattr(model.backbone, 'item_embeddings')
    values = {}
    size = config['localization']['batch_size']
    rule = config['global_localization']
    signature = hashlib.sha256()
    for name in ('request32', 'vectors', 'mask', 'h0', 'lengths'):
        if name in prepared['train']:
            value = prepared['train'][name].detach().cpu().contiguous().numpy()
            signature.update(str((name, value.shape, value.dtype)).encode())
            signature.update(value.tobytes())
    if not (folder / 'protocol.json').exists() and any(folder.glob('*.npy')):
        raise ValueError('Unbound localization cache: use a new folder for the learned projection')
    bind(folder, dict(version='learned_projection_v1', inputs_sha256=signature.hexdigest(),
                      blocks=blocks, selected_blocks=count, rule=rule))
    for split, inputs in prepared.items():
        if split != 'train':
            continue
        key = folder / f'causal_{split}.npy'
        if key.exists():
            score = np.load(key)
        else:
            chunks = []
            for start in range(0, len(inputs['h0']), size):
                stop = min(start + size, len(inputs['h0']))
                cache = folder / f'causal_{split}_{start:06d}.npy'
                if cache.exists():
                    result = np.load(cache)
                else:
                    x = {k: v[start:stop].to(config['device']) for (k, v) in inputs.items()}
                    generator = torch.Generator(device='cpu').manual_seed(rule['seed'] + start)
                    noise = torch.randn(rule['noise_repeats'], *x['vectors'].shape, generator=generator)
                    rms = (x['vectors'].square().sum((1, 2)) / (x['mask'].sum(1) * x['vectors'].shape[-1])).sqrt()
                    noise = noise.to(config['device']) * rms[None, :, None, None] * rule['noise_scale']
                    result = restoration_blocks(model, x['vectors'], x['lengths'], x['request32'], noise) if transfer else restoration_scores(model.backbone, x['vectors'], x['mask'], x['request32'], blocks, noise)
                    result = result.detach().cpu().numpy()
                    np.save(cache, result)
                if result.shape != (stop - start, len(blocks)) or not np.isfinite(result).all():
                    raise ValueError('Invalid localization cache shape or values')
                chunks.append(result)
            score = np.concatenate(chunks)
            np.save(key, score)
        if score.shape != (len(inputs['h0']), len(blocks)):
            raise ValueError('Localization user count differs')
        values[split] = score
        log.info('Localization %s %s: users=%d blocks=%d', 'causal', split, len(score), len(blocks))
    masks = {}
    mean = values['train'].astype(np.float64).mean(0) if transfer else values['train'].mean(0)
    for split, inputs in prepared.items():
        shape = (len(inputs['h0']), len(blocks))
        score = np.broadcast_to(mean, shape)
        chosen = np.argsort(-score, axis=1, kind='stable')[:, :count]
        mask = np.zeros(shape, dtype=np.float32)
        np.put_along_axis(mask, chosen, 1.0, axis=1)
        masks[split] = torch.from_numpy(mask)
        np.save(folder.parent / f'{split}_block_masks.npy', mask)
    write(folder / 'block_layout.json', blocks)
    write(folder / 'selection.json', dict(rule='training_mean_absolute_gaussian_restoration_effect', mean_scores=mean.tolist(), selected_blocks=np.argsort(-mean, kind='stable')[:count].tolist()))
    return masks

@torch.no_grad()
def project_requests(query, projection):
    """Map requests with the trained semantic MLP."""
    from .projection import SemanticProjection
    if not isinstance(projection, SemanticProjection):
        raise TypeError('Pass a trained SemanticProjection module')
    if not bool(projection.fitted) or any(p.requires_grad for p in projection.parameters()):
        raise ValueError('Fit and freeze the projection before computing localization directions')
    projected = projection(query)
    if not torch.isfinite(projected).all() or bool((projected.norm(dim=-1) < 1e-10).any()):
        raise ValueError('Request has a degenerate learned projection')
    return projected.to(query.device)

@contextmanager
def hooks(handles):
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

def _capture(backbone, vectors, mask, grad=False):
    (outputs, inputs, handles) = ({}, {}, [])
    for site in backbone.sites:

        def record(module, args, output, key=site.index):
            inputs[key] = args[0]
            outputs[key] = output
        handles.append(site.module.register_forward_hook(record))
    with hooks(handles), torch.set_grad_enabled(grad), backbone.edit(None):
        user = backbone.encode(vectors, mask)
    return (user, inputs, outputs)

def block_layout(backbone, weights_per_block):
    blocks = []
    for site in backbone.sites:
        m = site.module
        if weights_per_block % m.in_features:
            raise ValueError('Block budget must divide input width')
        rows = weights_per_block // m.in_features
        if m.out_features % rows:
            raise ValueError('Output rows must divide into equal-cost blocks')
        for start in range(0, m.out_features, rows):
            blocks.append(dict(site=site.index, layer=site.layer_index, matrix=site.matrix_name, start=start, stop=start + rows, weights=weights_per_block))
    return blocks

@torch.no_grad()
def restoration_scores(backbone, vectors, mask, request, blocks, noises):
    backbone.eval()
    (_, _, clean) = _capture(backbone, vectors, mask)
    repeats = []
    for noise in noises:
        corrupted = vectors + noise * mask[..., None]
        with backbone.edit(None):
            before = (backbone.encode(corrupted, mask) * request).sum(-1)
        effects = []
        for block in blocks:

            def patch(module, args, output):
                changed = output.clone()
                (a, z) = (block['start'], block['stop'])
                changed[..., a:z] = torch.where(mask[..., None], clean[block['site']][..., a:z], output[..., a:z])
                return changed
            handle = backbone.sites[block['site']].module.register_forward_hook(patch)
            try:
                with backbone.edit(None):
                    after = (backbone.encode(corrupted, mask) * request).sum(-1)
            finally:
                handle.remove()
            effects.append(after - before)
        repeats.append(torch.stack(effects, -1).abs())
    return torch.stack(repeats).mean(0)

@torch.no_grad()
def restoration_blocks(editor, vectors, lengths, request, noises):
    editor.eval()
    with editor.capture():
        editor.base_encode(vectors, lengths)
        clean = [[output.detach() for output in site.outputs] for (_, _, site) in editor.sites]
    mask = torch.arange(vectors.shape[1], device=vectors.device)[None] < lengths[:, None]
    repeats = []
    for noise in noises:
        corrupted = vectors + noise * mask[..., None]
        before = (editor.base_encode(corrupted, lengths) * request).sum(-1)
        effects = []
        for block in editor.blocks:
            with editor.restore_activation(block, clean[block['site']]):
                after = (editor.base_encode(corrupted, lengths) * request).sum(-1)
            effects.append((after - before).abs())
        repeats.append(torch.stack(effects, -1))
    return torch.stack(repeats).mean(0)
