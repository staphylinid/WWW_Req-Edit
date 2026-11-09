"""Shared training, checkpoint resume, candidate evaluation and protection evaluation."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from .loss import click_pairs, click_loss, editor_loss, finite_gradients
from .metrics import ranking_metrics, semantic_list_metrics
from .utils import bind, write, checkpoint, seed_all, trainable_state, load_trainable_state, frozen_parameters, frozen_digest

def batch(inputs, indices, device):
    return {k: v[indices].to(device) for (k, v) in inputs.items()}

def item_table(model):
    return model.backbone.item_embeddings if hasattr(model.backbone, 'item_embeddings') else model.backbone.table

def encode(model, x, mask):
    if hasattr(model, 'encode_batch'):
        return model.encode_batch(x, mask)
    if hasattr(model, 'encode_located'):
        (user, _, reg, _) = model.encode_located(x['vectors'], x['mask'], x['query'], x['h0'], mask)
        return (user, reg)
    return model.encode_edited(x['vectors'], x['lengths'], x['query'], x['h0'], mask)

def scores(model, user, ids):
    if hasattr(model.backbone, 'scores'):
        return model.backbone.scores(user, ids)
    return torch.einsum('bd,bnd->bn', user, torch.nn.functional.embedding(ids, item_table(model)))

@torch.no_grad()
def evaluate(model, inputs, masks, rows, pools, config, assets=None):
    model.eval()
    result = []
    size = config['training']['eval_batch_size']
    for start in range(0, len(rows), size):
        ix = slice(start, start + size)
        x = batch(inputs, ix, config['device'])
        user = x['h0'] if masks is None else encode(model, x, masks[ix].to(config['device']))[0]
        subset = rows[ix]
        width = max((len(pools[r['user_id']]) for r in subset))
        ids = torch.zeros(len(subset), width, dtype=torch.long, device=config['device'])
        for (j, row) in enumerate(subset):
            values = pools[row['user_id']]
            ids[j, :len(values)] = torch.as_tensor(values, device=config['device'])
        prediction = scores(model, user, ids).masked_fill(ids == 0, -torch.inf)
        order = prediction.argsort(dim=-1, descending=True, stable=True)[:, :20]
        result.append(ids.gather(1, order).cpu().numpy())
    ranks = np.concatenate(result)
    (metrics, detail) = ranking_metrics(ranks, rows, return_per_user=True)
    metrics.update(candidate_scope=config['evaluation'].get('candidate_scope', 'provided_candidates'), history_items_excluded_from_ranking=False)
    if assets is not None and getattr(assets, 'semantic', None) is not None:
        (extra, semantic_detail) = semantic_list_metrics(ranks, rows, assets.semantic, inputs.get('evaluation_query', inputs['query']).numpy(), categories=getattr(assets, 'categories', None), thresholds=config['evaluation']['thresholds'], return_per_user=True)
        metrics.update(extra)
        detail = [dict(a, **b) for (a, b) in zip(detail, semantic_detail)]
    return (metrics, detail, ranks)

def display(metrics):
    return ' '.join((f'{key}={metrics[key]:.4f}' for key in ('micro_recall@10', 'ndcg@10', 'micro_recall@20', 'ndcg@20') if key in metrics))

def train(model, prepared, masks, assets, config, seed, folder, log, method='causal', *, before_batch=None, regularization_weight=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        raise FileExistsError('Run already completed; use a new output directory')
    if not hasattr(model.backbone, 'item_embeddings'):
        seed_all(seed)
    if (folder / 'last.pt').exists() and not (folder / 'protocol.json').exists():
        raise ValueError('Cannot resume without a matching run protocol')
    protocol = hashlib.sha256()
    for split in ('train', 'valid', 'test'):
        protocol.update(split.encode())
        for name, value in sorted({**prepared[split], 'selected_blocks': masks[split]}.items()):
            array = value.detach().cpu().contiguous().numpy()
            protocol.update(str((name, array.shape, array.dtype)).encode())
            protocol.update(memoryview(array).cast('B'))
        for row in assets.data.rows[split]:
            values = [int(row['user_id']), list(map(int, row['history'])), list(map(int, row['positive']))]
            if split != 'train':
                values.append(list(map(int, assets.pools[split][row['user_id']])))
            protocol.update(json.dumps(values, separators=(',', ':')).encode())
    bind(folder, dict(input_sha256=protocol.hexdigest(), config=config, seed=seed,
                      method=method, regularization_weight=regularization_weight,
                      base_sha256=assets.provenance['base_sha256'], catalog_size=int(assets.data.n)))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config['training']['lr'], foreach=False)
    original = frozen_digest(model)
    versions = [p._version for p in frozen_parameters(model)]
    t = config['training']
    (first, best, bad, elapsed, history) = (0, -1.0, 0, 0.0, [])
    if (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if saved['base_sha256'] != assets.provenance['base_sha256']:
            raise ValueError('Resume backbone differs')
        load_trainable_state(model, saved['editor'])
        optimizer.load_state_dict(saved['optimizer'])
        (first, best, bad, elapsed, history) = (saved['epoch'], saved['best'], saved['bad'], saved.get('elapsed_seconds', 0.0), saved['history'])
        torch.set_rng_state(saved['rng_cpu'])
        if saved.get('rng_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
        log.info('Resume after epoch %d', first)
    started = time.perf_counter()
    for epoch in range(first, t['epochs']):
        if epoch >= t['min_epochs'] and bad >= t['patience']:
            break
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(assets.data.rows['train']))
        totals = torch.zeros(3, dtype=torch.float64, device=config['device'])
        count = 0
        tick = time.perf_counter()
        for offset in range(0, len(order), t['batch_size']):
            if before_batch is not None:
                before_batch(epoch, offset, len(order))
            ix = order[offset:offset + t['batch_size']]
            rows = [assets.data.rows['train'][i] for i in ix]
            x = batch(prepared['train'], ix, config['device'])
            (user, regularizer) = encode(model, x, masks['train'][ix].to(config['device']))
            (owner, positive, negative) = click_pairs(rows, assets.data.n, rng, config['device'])
            scoring = model.backbone.scores if hasattr(model.backbone, 'scores') else None
            click = click_loss(user, item_table(model), owner, positive, negative, scoring)
            weight = t['regularization'] if regularization_weight is None else regularization_weight
            loss = editor_loss(click, regularizer, weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not finite_gradients(parameters):
                raise FloatingPointError('Nonfinite editor gradient')
            optimizer.step()
            totals.add_(torch.stack((loss.detach(), click.detach(), regularizer.detach())).to(torch.float64) * len(positive))
            count += len(positive)
        if versions != [p._version for p in frozen_parameters(model)]:
            raise RuntimeError('Frozen backbone changed')
        (metrics, _, _) = evaluate(model, prepared['valid'], masks['valid'], assets.data.rows['valid'], assets.pools['valid'], config)
        value = metrics[config['evaluation']['selection_metric']]
        improved = value > best
        (best, bad) = (value, 0) if improved else (best, bad + 1)
        sums = totals.cpu().numpy() / count
        history.append(dict(epoch=epoch + 1, loss=float(sums[0]), click_loss=float(sums[1]), editor_reg=float(sums[2]), validation=metrics, seconds=time.perf_counter() - tick, improved=improved, bad_epochs=bad))
        state = dict(editor=trainable_state(model), optimizer=optimizer.state_dict(), epoch=epoch + 1, best=best, bad=bad, history=history, elapsed_seconds=elapsed + time.perf_counter() - started, base_sha256=assets.provenance['base_sha256'], rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if str(config['device']).startswith('cuda') else None)
        checkpoint(folder / 'last.pt', state)
        if improved:
            checkpoint(folder / 'best.pt', state)
        write(folder / 'history.json', history)
        log.info('%s seed=%d epoch=%d loss=%.4f %s best=%.4f bad=%d %s', method, seed, epoch + 1, sums[0], display(metrics), best, bad, 'BEST' if improved else '')
    saved = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    load_trainable_state(model, saved['editor'])
    result = dict(method=method, seed=seed, best_epoch=saved['epoch'], trained_epochs=len(history), elapsed_seconds=elapsed + time.perf_counter() - started)
    for split in ('valid', 'test'):
        (metrics, per, ranks) = evaluate(model, prepared[split], masks[split], assets.data.rows[split], assets.pools[split], config, assets)
        result[split] = metrics
        write(folder / f'{split}_per_user.json', per)
        np.save(folder / f'{split}_top20.npy', ranks)
        log.info('%s %s', split, display(metrics))
    if frozen_digest(model) != original:
        raise RuntimeError('Frozen backbone checksum changed')
    result['frozen_base_unchanged'] = True
    write(folder / 'complete.json', result)
    return result
