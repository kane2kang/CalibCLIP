#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation utilities for retrieval tasks.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union


class Evaluator:
    """
    Evaluator for text-image retrieval tasks.
    """

    def __init__(
            self,
            topk: List[int] = [1, 5, 10],
            compute_map: bool = True,
            compute_minp: bool = True,
    ):
        """
        Args:
            topk: List of k values for Recall@k
            compute_map: Whether to compute mAP
            compute_minp: Whether to compute mINP
        """
        self.topk = topk
        self.compute_map = compute_map
        self.compute_minp = compute_minp

    def evaluate(
            self,
            similarity_matrix: torch.Tensor,
            ground_truth: Dict[int, List[int]],
            query_type: str = "text",
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            similarity_matrix: Similarity scores (num_queries, num_gallery)
            ground_truth: Dict mapping query index to list of positive gallery indices
            query_type: "text" or "image"

        Returns:
            Dictionary of metrics
        """
        if isinstance(similarity_matrix, torch.Tensor):
            similarity_matrix = similarity_matrix.cpu().numpy()

        num_queries = similarity_matrix.shape[0]

        # Get ranking for each query
        rankings = np.argsort(-similarity_matrix, axis=1)

        metrics = {}

        # Compute Recall@k
        for k in self.topk:
            recall = self._compute_recall_at_k(rankings, ground_truth, k)
            metrics[f"R@{k}"] = recall

        # Compute mAP
        if self.compute_map:
            map_score = self._compute_map(rankings, ground_truth)
            metrics["mAP"] = map_score

        # Compute mINP
        if self.compute_minp:
            minp_score = self._compute_minp(rankings, ground_truth)
            metrics["mINP"] = minp_score

        return metrics

    def _compute_recall_at_k(
            self,
            rankings: np.ndarray,
            ground_truth: Dict[int, List[int]],
            k: int,
    ) -> float:
        """Compute Recall@k."""
        num_queries = len(ground_truth)
        hits = 0

        for query_idx, positive_indices in ground_truth.items():
            if query_idx >= rankings.shape[0]:
                continue
            top_k_predictions = set(rankings[query_idx, :k].tolist())
            if len(set(positive_indices) & top_k_predictions) > 0:
                hits += 1

        return hits / num_queries if num_queries > 0 else 0.0

    def _compute_map(
            self,
            rankings: np.ndarray,
            ground_truth: Dict[int, List[int]],
    ) -> float:
        """Compute mean Average Precision."""
        aps = []

        for query_idx, positive_indices in ground_truth.items():
            if query_idx >= rankings.shape[0]:
                continue

            positive_set = set(positive_indices)
            num_positives = len(positive_set)

            if num_positives == 0:
                continue

            ranking = rankings[query_idx]
            ap = 0.0
            num_hits = 0

            for rank, gallery_idx in enumerate(ranking):
                if gallery_idx in positive_set:
                    num_hits += 1
                    precision_at_rank = num_hits / (rank + 1)
                    ap += precision_at_rank

            ap /= num_positives
            aps.append(ap)

        return np.mean(aps) if len(aps) > 0 else 0.0

    def _compute_minp(
            self,
            rankings: np.ndarray,
            ground_truth: Dict[int, List[int]],
    ) -> float:
        """
        Compute mean Inverse Normalized Precision (mINP).

        mINP measures the hardest positive retrieval performance.
        """
        inps = []

        for query_idx, positive_indices in ground_truth.items():
            if query_idx >= rankings.shape[0]:
                continue

            positive_set = set(positive_indices)
            num_positives = len(positive_set)

            if num_positives == 0:
                continue

            ranking = rankings[query_idx]

            # Find the rank of the hardest positive (last positive in ranking)
            positive_ranks = []
            for rank, gallery_idx in enumerate(ranking):
                if gallery_idx in positive_set:
                    positive_ranks.append(rank)

            if len(positive_ranks) == 0:
                continue

            hardest_rank = max(positive_ranks)  # 0-indexed

            # INP = (num_positives) / (hardest_rank + 1 - num_negatives_before_hardest)
            # Simplified: INP = num_positives / (hardest_rank + 1)
            inp = num_positives / (hardest_rank + 1)
            inps.append(inp)

        return np.mean(inps) if len(inps) > 0 else 0.0


def compute_similarity(
        query_features: torch.Tensor,
        gallery_features: torch.Tensor,
        normalize: bool = True,
) -> torch.Tensor:
    """
    Compute cosine similarity between query and gallery features.

    Args:
        query_features: Query features (N, D)
        gallery_features: Gallery features (M, D)
        normalize: Whether to L2 normalize features

    Returns:
        Similarity matrix (N, M)
    """
    if normalize:
        query_features = torch.nn.functional.normalize(query_features, p=2, dim=-1)
        gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=-1)

    return query_features @ gallery_features.t()
