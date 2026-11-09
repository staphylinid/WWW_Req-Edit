from __future__ import annotations

import math
from collections import Counter
import numpy as np

DEFAULT_THRESHOLDS = {
    "history_low_coverage_max_cos": 0.60,
    "query_relevance_min_cos": 0.25,
    "near_duplicate_min_cos": 0.95,
    "calibration_status": "exploratory_uncalibrated",
}

def _ids(row, key, maximum=None):
    values = np.asarray(row[key], dtype=np.int64).reshape(-1)
    values = np.unique(values[values != 0])
    if np.any(values < 1) or (maximum is not None and np.any(values > maximum)):
        raise ValueError(f"{key} contains out-of-catalog item indices")
    return values

def _validate_rankings(rank_ids, rows):
    ranks = np.asarray(rank_ids, dtype=np.int64)
    if ranks.ndim != 2 or len(ranks) != len(rows) or not len(rows):
        raise ValueError("Nonempty rank_ids and rows must match")
    if np.any(ranks < 0):
        raise ValueError("Negative item index")
    for rank in ranks:
        nonzero = rank[rank != 0]
        if len(nonzero) != len(np.unique(nonzero)):
            raise ValueError("Ranking contains duplicate item IDs")
    return ranks

def ranking_metrics(rank_ids, rows, ks=(5, 10, 20), *, return_per_user=False):
    ranks = _validate_rankings(rank_ids, rows)
    truths = [set(_ids(row, "positive").tolist()) for row in rows]
    if any(not truth for truth in truths):
        raise ValueError("Every evaluated user requires positives")
    total_positive = sum(map(len, truths))
    per_user = [{"user_id": (row.get("user_id").item() if isinstance(row.get("user_id"), np.generic)
                             else row.get("user_id")), "positive_items": len(truth)}
                for row, truth in zip(rows, truths)]
    result = {"users": len(rows), "positive_items": total_positive,
              "aggregation": "micro_click_and_macro_user", "padding_item_id": 0}
    for k in ks:
        if k < 1:
            raise ValueError("Cutoffs must be positive")
        recalls, ndcgs, hit_counts = [], [], []
        for index, (rank, truth) in enumerate(zip(ranks, truths)):
            hits = np.asarray([int(item in truth) for item in rank[:k]], dtype=np.float64)
            hit_counts.append(int(hits.sum()))
            recalls.append(hits.sum() / len(truth))
            ideal = sum(1 / math.log2(position + 2) for position in range(min(k, len(truth))))
            ndcgs.append(float((hits / np.log2(np.arange(len(hits)) + 2)).sum() / ideal))
            per_user[index].update({f"hits@{k}": hit_counts[-1], f"recall@{k}": float(recalls[-1]),
                                    f"ndcg@{k}": ndcgs[-1]})
        result.update({f"recall@{k}": float(np.mean(recalls)),
                       f"macro_recall@{k}": float(np.mean(recalls)),
                       f"micro_recall@{k}": sum(hit_counts) / total_positive,
                       f"ndcg@{k}": float(np.mean(ndcgs)),
                       f"hit_rate@{k}": float(np.mean(np.asarray(hit_counts) > 0))})
    return (result, per_user) if return_per_user else result

def _normalize(vectors):
    values = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(values).all(axis=-1) & np.isfinite(lengths) & (lengths > 1e-12)
    normalized = np.zeros_like(values)
    normalized[valid] = values[valid] / lengths[valid, None]
    return normalized, valid

def _mean(values):
    return float(np.mean(values)) if values else None

def semantic_list_metrics(rank_ids, rows, semantic_items, query_vectors=None, *,
                          thresholds=None, categories=None, ks=(5, 10, 20), return_per_user=False):

    ranks = _validate_rankings(rank_ids, rows)
    if len(semantic_items.shape) != 2 or semantic_items.shape[1] != 1024:
        raise ValueError("Semantic proxies require the original fixed 1024-D item vectors")
    if query_vectors is not None and tuple(query_vectors.shape) != (len(rows), 1024):
        raise ValueError("query_vectors must be aligned [users,1024]")
    catalog = semantic_items.shape[0] - 1
    if np.any(ranks > catalog):
        raise ValueError("Ranking item index exceeds semantic table")
    config = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    for field in ("history_low_coverage_max_cos", "query_relevance_min_cos", "near_duplicate_min_cos"):
        if not -1 <= float(config[field]) <= 1:
            raise ValueError(f"Invalid cosine threshold: {field}")
    if config["query_relevance_min_cos"] >= 1:
        raise ValueError("query_relevance_min_cos must be below 1 for linear relevance weights")
    ks = tuple(int(k) for k in ks)
    if any(k < 1 for k in ks):
        raise ValueError("Cutoffs must be positive")
    buckets = {k: {} for k in ks}
    low_hits = {k: 0 for k in ks}
    low_positives = 0
    qualifying_users = 0
    known_low_users = 0
    counts = {"users": len(rows), "users_with_valid_query": 0,
              "users_with_complete_nonempty_history_semantics": 0,
              "topk_valid_semantic_items": 0, "topk_items": 0}
    names = ("low_coverage_positive_macro_recall", "semantic_item_coverage", "ILD",
             "rank_rel_weighted_ILD_proxy", "near_duplicate_slot_rate_proxy", "request_cosine_mean_proxy",
             "nonhit_low_request_relevance_rate_proxy", "dedup_relevant_low_coverage_count_per_k_proxy",
             "category_known_item_fraction", "category_entropy_bits", "category_unique_count", "category_unique_count_per_k")
    per_user = []
    current_record = None

    def add(k, name, value):
        if value is not None:
            buckets[k].setdefault(name, []).append(float(value))
            current_record[f"{name}@{k}"] = float(value)

    for index, (rank, row) in enumerate(zip(ranks, rows)):
        user_id = row.get("user_id")
        current_record = {"user_id": user_id.item() if isinstance(user_id, np.generic) else user_id,
                          **{f"{name}@{k}": None for k in ks for name in names}}
        history = _ids(row, "history", catalog)
        positives = _ids(row, "positive", catalog)
        selected_ids = np.unique(np.concatenate((rank[rank > 0], history, positives)))
        
        vectors, valid = _normalize(semantic_items[selected_ids])
        lookup = {int(item): j for j, item in enumerate(selected_ids)}
        h_indices = np.asarray([lookup[int(item)] for item in history], dtype=np.int64)
        history_known = bool(len(history) and valid[h_indices].all())
        counts["users_with_complete_nonempty_history_semantics"] += int(history_known)
        h_max = np.full(len(selected_ids), np.nan, dtype=np.float32)
        if history_known:
            h_max[valid] = np.clip(vectors[valid] @ vectors[h_indices].T, -1, 1).max(axis=1)
        p_indices = np.asarray([lookup[int(item)] for item in positives], dtype=np.int64)
        positives_known = bool(history_known and len(positives) and valid[p_indices].all())
        known_low_users += int(positives_known)
        low_truth = ({int(item) for item, j in zip(positives, p_indices)
                      if h_max[j] < config["history_low_coverage_max_cos"]} if positives_known else set())
        qualifying_users += int(bool(low_truth))
        low_positives += len(low_truth)
        q_valid = False
        q_cos = np.full(len(selected_ids), np.nan, dtype=np.float32)
        if query_vectors is not None:
            query, is_valid = _normalize(np.asarray(query_vectors[index:index + 1]))
            q_valid = bool(is_valid[0])
            if q_valid:
                q_cos[valid] = np.clip(vectors[valid] @ query[0], -1, 1)
        counts["users_with_valid_query"] += int(q_valid)
        current_record.update(history_distinct_items=len(history), history_semantics_complete=history_known,
                              query_semantics_valid=q_valid, low_coverage_positive_labels_known=positives_known,
                              low_coverage_positive_ids=sorted(low_truth), low_coverage_positive_items=len(low_truth),
                              subjective_satisfaction=None)
        if return_per_user:
            current_record["rank_item_proxies"] = [
                {"item_id": int(item), "semantic_available": bool(valid[lookup[int(item)]]),
                 "history_max_cos": float(h_max[lookup[int(item)]]) if np.isfinite(h_max[lookup[int(item)]]) else None,
                 "query_cos": float(q_cos[lookup[int(item)]]) if np.isfinite(q_cos[lookup[int(item)]]) else None}
                for item in rank if item]
        counts["topk_items"] += int(np.count_nonzero(rank))
        counts["topk_valid_semantic_items"] += sum(bool(valid[lookup[int(item)]]) for item in rank if item)
        for k in ks:
            items = rank[:k]
            items = items[items > 0]
            ids = np.asarray([lookup[int(item)] for item in items], dtype=np.int64)
            semantic_ok = valid[ids]
            known_ids = ids[semantic_ok]
            full_semantics = bool(len(items) and semantic_ok.all())
            if low_truth:
                hits = len(set(map(int, items)) & low_truth)
                add(k, "low_coverage_positive_macro_recall", hits / len(low_truth))
                low_hits[k] += hits
                current_record[f"low_coverage_positive_hits@{k}"] = hits
            else:
                current_record[f"low_coverage_positive_hits@{k}"] = None
            add(k, "semantic_item_coverage", len(known_ids) / len(items) if len(items) else None)
            if len(known_ids) > 1:
                similarities = np.clip(vectors[known_ids] @ vectors[known_ids].T, -1, 1)
                upper = np.triu_indices(len(known_ids), 1)
                distances = 1 - similarities[upper]
                add(k, "ILD", float(distances.mean()))
                if q_valid:
                    
                    discounts = 1 / np.log2(np.flatnonzero(semantic_ok) + 2)
                    relevance = np.clip((q_cos[known_ids] - config["query_relevance_min_cos"]) /
                                        (1 - config["query_relevance_min_cos"]), 0, 1)
                    weights = discounts * relevance
                    pair_weights = weights[upper[0]] * weights[upper[1]]
                    if pair_weights.sum() > 0:
                        add(k, "rank_rel_weighted_ILD_proxy", float(np.dot(distances, pair_weights) / pair_weights.sum()))
            if full_semantics:
                similarities = np.clip(vectors[ids] @ vectors[ids].T, -1, 1)
                duplicate_slots = sum(bool(np.any(similarities[j, :j] >= config["near_duplicate_min_cos"]))
                                      for j in range(1, len(ids)))
                add(k, "near_duplicate_slot_rate_proxy", duplicate_slots / k)
            if q_valid and len(known_ids):
                add(k, "request_cosine_mean_proxy", float(q_cos[known_ids].mean()))
                nonhits = [lookup[int(item)] for item in items
                           if item not in set(positives) and valid[lookup[int(item)]]]
                if nonhits:
                    add(k, "nonhit_low_request_relevance_rate_proxy",
                        float(np.mean(q_cos[nonhits] < config["query_relevance_min_cos"])))
            if full_semantics and history_known and q_valid:
                accepted = []
                for j in ids:  
                    if q_cos[j] < config["query_relevance_min_cos"] or h_max[j] >= config["history_low_coverage_max_cos"]:
                        continue
                    if not accepted or np.all(vectors[accepted] @ vectors[j] < config["near_duplicate_min_cos"]):
                        accepted.append(j)
                add(k, "dedup_relevant_low_coverage_count_per_k_proxy", len(accepted) / k)
            if categories is not None:
                labels = []
                for item in items:
                    label = categories.get(int(item)) if isinstance(categories, dict) else categories[int(item)]
                    if label is None or str(label).strip() == "" or isinstance(label, (float, np.floating)) and not math.isfinite(label):
                        continue
                    labels.append(str(label))
                add(k, "category_known_item_fraction", len(labels) / len(items) if len(items) else None)
                if labels:
                    frequencies = np.asarray(list(Counter(labels).values()), dtype=np.float64) / len(labels)
                    add(k, "category_entropy_bits", float(-np.sum(frequencies * np.log2(frequencies))))
                    add(k, "category_unique_count", len(frequencies))
                    add(k, "category_unique_count_per_k", len(frequencies) / k)
        if return_per_user:
            per_user.append(current_record)

    result = {"semantic_threshold_config": config, "semantic_metrics_status": "exploratory_proxies",
              "semantic_space": "bge_m3_1024d_cosine", "semantic_counts": counts,
              "low_coverage_positive_eligible_users": qualifying_users,
              "low_coverage_positive_label_known_users": known_low_users,
              "low_coverage_positive_items": low_positives, "subjective_satisfaction": None,
              "semantic_definitions": {
                  "history_low_coverage": "max cosine to all distinct supplied history items < threshold; empty/incomplete history is unknown",
                  "low_coverage_recall": "macro only over users with >=1 qualifying real positive; micro denominator qualifying positives",
                  "request_relevance": "cosine proxy, not independently measured request satisfaction",
                  "nonhit": "not a recorded target positive; not a verified irrelevant/negative item",
                  "near_duplicate_slot_rate": "rank slot duplicates any earlier item at cosine >= threshold; count/requested K",
                  "ILD": "mean pairwise (1-cosine), range [0,2], among valid semantic vectors",
                  "rank_rel_weighted_ILD_proxy": "pairwise (1-cosine) weighted by products of [1/log2(rank+1)] * clipped((qcos-threshold)/(1-threshold)); normalized by pair-weight sum; not exact EILD",
                  "dedup_relevant_low_coverage": "rank-greedy semantic dedup of qcos>=threshold and history maxcos<threshold items; count/requested K",
                  "category": "single supplied high-level label; entropy over known labels only; unique_count_per_k is not catalog-taxonomy coverage",
                  "aggregation": "macro users with defined metric; each *_users reports denominator; zero-support metrics are null"}}
    for k in ks:
        for name in names:
            values = buckets[k].get(name, [])
            result[f"{name}@{k}"] = _mean(values)
            result[f"{name}_users@{k}"] = len(values)
        result[f"low_coverage_positive_micro_recall@{k}"] = low_hits[k] / low_positives if low_positives else None
        result[f"low_coverage_positive_hits@{k}"] = low_hits[k]
    return (result, per_user) if return_per_user else result
