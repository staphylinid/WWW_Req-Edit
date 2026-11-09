"""Request-compatible history selection and inference-time update projection."""
import numpy as np
import torch
from contextlib import contextmanager


@contextmanager
def protected_history(model, row, query, semantic, threshold, eta, vectors, mask_or_lengths):
    """Capture unedited internal inputs and protect their selected history directions."""
    transfer=not hasattr(model.backbone,'item_embeddings')
    sites=[s[2] for s in model.sites] if transfer else [s.module for s in model.backbone.sites]
    positions,protected,heldout=choose_history(row,query,semantic,threshold)
    previous=[(getattr(s,'protection_eta',0.),getattr(s,'protection_basis',None)) for s in sites]
    records=[[] for _ in sites]
    handles=[]
    try:
        for site in sites:site.protection_eta=0.
        if positions and eta:
            for index,site in enumerate(sites):
                def capture(module,inputs,j=index):
                    records[j].append(inputs[0].reshape(1,-1,module.in_features))
                handles.append(site.register_forward_pre_hook(capture))
            if transfer:model.base_encode(vectors,mask_or_lengths)
            else:
                with model.backbone.edit(None):model.backbone.encode(vectors,mask_or_lengths)
            for handle in handles:handle.remove()
            handles=[]
            for site,parts in zip(sites,records):
                internal=torch.cat(parts,dim=1)[0]
                site.protection_basis=orthogonal_basis(internal[positions].T)[None]
                site.protection_eta=eta
        yield dict(positions=positions,protected=protected,heldout=heldout)
    finally:
        for handle in handles:handle.remove()
        for site,(strength,basis) in zip(sites,previous):
            site.protection_eta=strength
            site.protection_basis=basis


@torch.inference_mode()
def evaluate_protection(model,prepared,masks,assets,config,folder,eta,log):
    from pathlib import Path
    import time
    from .train import encode,scores,item_table
    from .metrics import ranking_metrics,semantic_list_metrics
    from .utils import write,frozen_digest
    if not 0<=eta<=1:raise ValueError('Protection eta must be in [0,1]')
    folder=Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        raise FileExistsError('Evaluation already completed; use a new output directory')
    threshold=calibrate(assets.data.rows['train'],assets.queries,assets.semantic)
    before=frozen_digest(model)
    model.eval()
    reports={}
    transfer=not hasattr(model.backbone,'item_embeddings')
    for split in ('valid','test'):
        rows=assets.data.rows[split]
        rankings=[];details=[];drifts=[];times=[]
        for index,row in enumerate(rows):
            if str(config['device']).startswith('cuda'):torch.cuda.synchronize()
            started=time.perf_counter()
            x={k:v[index:index+1].to(config['device']) for k,v in prepared[split].items()}
            query=np.asarray(assets.queries[row['query_row']],dtype=np.float32)
            selected=masks[split][index:index+1].to(config['device'])
            with protected_history(model,row,query,assets.semantic,threshold,eta,x['vectors'],
                                   x['lengths'] if transfer else x['mask']) as keep:
                user,_=encode(model,x,selected)
            ids=torch.tensor(assets.pools[split][row['user_id']],device=config['device'])
            score=scores(model,user,ids[None])[0]
            rank=ids[score.argsort(descending=True,stable=True)[:20]]
            if str(config['device']).startswith('cuda'):torch.cuda.synchronize()
            times.append((time.perf_counter()-started)*1000)
            rankings.append(rank.cpu().numpy())
            detail=dict(user_id=row['user_id'],selected_packed_positions=keep['positions'],
                        protected_catalog_ids=keep['protected'],heldout_catalog_ids=keep['heldout'])
            if keep['positions'] and keep['heldout']:
                heldout=torch.tensor(keep['heldout'],device=config['device'])[None]
                difference=(scores(model,user,heldout)-scores(model,x['h0'],heldout)).abs().mean()
                standard=scores(model,x['h0'],ids[None])[0].std().clamp_min(1e-8)
                drift=float(difference/standard);drifts.append(drift)
                detail['heldout_history_score_drift']=drift
            details.append(detail)
        rank=np.stack(rankings)
        metrics,per=ranking_metrics(rank,rows,return_per_user=True)
        extra,semantic=semantic_list_metrics(rank,rows,assets.semantic,
            prepared[split]['query'].numpy(),categories=assets.categories,
            thresholds=config['evaluation']['thresholds'],return_per_user=True)
        metrics.update(extra)
        metrics.update(candidate_scope=config['evaluation'].get('candidate_scope','provided_candidates'),history_items_excluded_from_ranking=False,
                       heldout_history_score_drift=float(np.mean(drifts)) if drifts else None)
        reports[split]=dict(metrics=metrics,eta=eta,threshold=threshold,
            latency_mean_ms=float(np.mean(times)),latency_median_ms=float(np.median(times)),
            latency_p95_ms=float(np.quantile(times,.95)),protected_users=sum(bool(v['selected_packed_positions']) for v in details),
            timing_boundary='cached text vectors and prepared base histories; input transfer, frozen capture, history selection, SVD, edited forward and candidate ranking')
        np.save(folder/f'{split}_top20.npy',rank)
        write(folder/f'{split}_per_user.json',[{**a,**b,**c} for a,b,c in zip(per,semantic,details)])
        log.info('Protection eta=%.2f %s R10=%.4f N10=%.4f',eta,split,metrics['micro_recall@10'],metrics['ndcg@10'])
    if frozen_digest(model)!=before:raise ValueError('Frozen parameters changed during protection')
    result=dict(eta=eta,reports=reports,frozen_base_unchanged=True)
    write(folder/'complete.json',result)
    return result
def choose_history(row, query, semantic, threshold):
    history = np.asarray(semantic[row['history']], dtype=np.float32)
    cos = history @ query / (np.linalg.norm(history, axis=1) * np.linalg.norm(query) + 1e-12)
    groups = np.asarray(row['groups'])
    unique = np.unique(groups)
    scores = np.asarray([cos[groups == group].max() for group in unique])
    positions = [int(i) for i in np.argsort(-scores, kind='stable') if scores[i] >= threshold][:2]
    positions = positions[:max(0, len(unique) - 1)]
    selected = {unique[i] for i in positions}
    protected = [int(item) for (item, group) in zip(row['history'], groups) if group in selected]
    heldout = [int(item) for (item, group) in zip(row['history'], groups) if group not in selected]
    return (positions, protected, heldout)

def orthogonal_basis(keys):
    if keys.shape[-1] == 0:
        return keys
    (u, s, _) = torch.linalg.svd(keys, full_matrices=False)
    return u[:, s > s.max().clamp_min(1e-12) * 1e-05]

def soft_project(delta, basis, eta):
    return delta - eta * (delta @ basis) @ basis.transpose(-1, -2)

def calibrate(train, queries, semantic):
    rng = np.random.default_rng(20261010)
    null = []
    for _ in range(4096):
        a = int(rng.integers(len(train)))
        b = int(rng.integers(len(train) - 1))
        b += b >= a
        row = train[b]
        item = int(row['history'][rng.integers(len(row['history']))])
        q = np.asarray(queries[train[a]['query_row']], dtype=np.float32)
        h = np.asarray(semantic[item], dtype=np.float32)
        null.append(float(q @ h / (np.linalg.norm(q) * np.linalg.norm(h) + 1e-12)))
    return float(np.quantile(null, 0.95))
