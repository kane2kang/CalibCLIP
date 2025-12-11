#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for retrieval.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_rank(
        similarity: np.ndarray,
        query_pids: np.ndarray,
        gallery_pids: np.ndarray,
        topk: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute Rank-K accuracy.

    Args:
        similarity: Similarity matrix [num_queries, num_gallery]
        query_pids: Query person IDs [num_queries]
        gallery_pids: Gallery person IDs [num_gallery]
        topk: List of K values

    Returns:
        results: Dictionary with Rank-K accuracies
    """
    num_queries = similarity.shape[0]

    # Get ranking indices (descending similarity)
    indices = np.argsort(-similarity, axis=1)

    # Get predicted PIDs
    pred_pids = gallery_pids[indices]

    # Compute matches
    matches = (pred_pids == query_pids[:, np.newaxis])

    results = {}
    for k in topk:
        # Check if correct match in top-k
        correct = matches[:, :k].any(axis=1)
        results[f"R@{k}"] = correct.mean() * 100

    return results


def compute_ap(
        similarity: np.ndarray,
        query_pid: int,
        gallery_pids: np.ndarray,
) -> float:
    """
    Compute Average Precision for a single query.

    Args:
        similarity: Similarity scores [num_gallery]
        query_pid: Query person ID
        gallery_pids: Gallery person IDs [num_gallery]

    Returns:
        ap: Average Precision
    """
    # Get ranking indices
    indices = np.argsort(-similarity)

    # Get matches
    matches = (gallery_pids[indices] == query_pid)

    if matches.sum() == 0:
        return 0.0

    # Compute precision at each recall point
    cum_matches = np.cumsum(matches)
    precision = cum_matches / (np.arange(len(matches)) + 1)

    # AP = mean precision at match positions
    ap = (precision * matches).sum() / matches.sum()

    return ap


def compute_map(
        similarity: np.ndarray,
        query_pids: np.ndarray,
        gallery_pids: np.ndarray,
) -> float:
    """
    Compute mean Average Precision (mAP).

    Args:
        similarity: Similarity matrix [num_queries, num_gallery]
        query_pids: Query person IDs [num_queries]
        gallery_pids: Gallery person IDs [num_gallery]

    Returns:
        mAP: Mean Average Precision
    """
    num_queries = similarity.shape[0]

    aps = []
    for i in range(num_queries):
        ap = compute_ap(similarity[i], query_pids[i], gallery_pids)
        aps.append(ap)

    return np.mean(aps) * 100


def compute_minp(
        similarity: np.ndarray,
        query_pid: int,
        gallery_pids: np.ndarray,
) -> float:
    """
    Compute Inverse Negative Penalty (mINP) for a single query.

    Args:
        similarity: Similarity scores [num_gallery]
        query_pid: Query person ID
        gallery_pids: Gallery person IDs [num_gallery]

    Returns:
        inp: Inverse Negative Penalty
    """
    indices = np.argsort(-similarity)
    matches = (gallery_pids[indices] == query_pid)

    if matches.sum() == 0:
        return 0.0

    # Find position of hardest positive (last match)
    match_positions = np.where(matches)[0]
    hardest_pos = match_positions[-1]

    # Number of positives
    num_pos = matches.sum()

    # INP = num_pos / (hardest_pos + 1)
    inp = num_pos / (hardest_pos + 1)

    return inp


def compute_minp_all(
        similarity: np.ndarray,
        query_pids: np.ndarray,
        gallery_pids: np.ndarray,
) -> float:
    """
    Compute mean Inverse Negative Penalty (mINP).

    Args:
        similarity: Similarity matrix [num_queries, num_gallery]
        query_pids: Query person IDs [num_queries]
        gallery_pids: Gallery person IDs [num_gallery]

    Returns:
        mINP: Mean Inverse Negative Penalty
    """
    num_queries = similarity.shape[0]

    inps = []
    for i in range(num_queries):
        inp = compute_minp(similarity[i], query_pids[i], gallery_pids)
        inps.append(inp)

    return np.mean(inps) * 100


def compute_metrics(
        similarity: np.ndarray,
        query_pids: np.ndarray,
        gallery_pids: np.ndarray,
        topk: List[int] = [1, 5, 10],
        compute_map_flag: bool = True,
        compute_minp_flag: bool = True,
) -> Dict[str, float]:
    """
    Compute all retrieval metrics.

    Args:
        similarity: Similarity matrix [num_queries, num_gallery]
        query_pids: Query person IDs [num_queries]
        gallery_pids: Gallery person IDs [num_gallery]
        topk: List of K values for Rank-K
        compute_map_flag: Whether to compute mAP
        compute_minp_flag: Whether to compute mINP

    Returns:
        metrics: Dictionary with all metrics
    """
    metrics = {}

    # Rank-K
    rank_metrics = compute_rank(similarity, query_pids, gallery_pids, topk)
    metrics.update(rank_metrics)

    # mAP
    if compute_map_flag:
        metrics["mAP"] = compute_map(similarity, query_pids, gallery_pids)

    # mINP
    if compute_minp_flag:
        metrics["mINP"] = compute_minp_all(similarity, query_pids, gallery_pids)

    return metrics
