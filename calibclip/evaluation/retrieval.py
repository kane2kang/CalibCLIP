#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retrieval engine for CalibCLIP.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class RetrievalEngine:
    """
    Retrieval engine supporting standard and two-stage retrieval.

    Args:
        lambda_weight: Weight for combining original and discriminative similarities
        top_k: Number of candidates for re-ranking (if using two-stage)
    """

    def __init__(
            self,
            lambda_weight: float = 0.5,
            top_k: int = 100,
    ):
        self.lambda_weight = lambda_weight
        self.top_k = top_k

    def compute_similarity(
            self,
            query_features: torch.Tensor,
            gallery_features: torch.Tensor,
            normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute similarity between queries and gallery.

        Args:
            query_features: Query features [Q, D]
            gallery_features: Gallery features [G, D]
            normalize: Whether to normalize features

        Returns:
            similarity: Similarity matrix [Q, G]
        """
        if normalize:
            query_features = F.normalize(query_features, dim=-1)
            gallery_features = F.normalize(gallery_features, dim=-1)

        similarity = torch.matmul(query_features, gallery_features.t())
        return similarity

    def two_stage_retrieval(
            self,
            query_features: torch.Tensor,
            query_disc_features: torch.Tensor,
            gallery_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Two-stage retrieval with re-ranking.

        Stage 1: Initial retrieval with original features
        Stage 2: Re-rank top-k with discriminative features

        Args:
            query_features: Original query features [Q, D]
            query_disc_features: Discriminative query features [Q, D]
            gallery_features: Gallery features [G, D]

        Returns:
            final_similarity: Combined similarity matrix [Q, G]
        """
        # Normalize all features
        query_features = F.normalize(query_features, dim=-1)
        query_disc_features = F.normalize(query_disc_features, dim=-1)
        gallery_features = F.normalize(gallery_features, dim=-1)

        # Stage 1: Initial similarity
        initial_sim = torch.matmul(query_features, gallery_features.t())

        # Stage 2: Discriminative similarity
        disc_sim = torch.matmul(query_disc_features, gallery_features.t())

        # Combine
        final_similarity = (
                self.lambda_weight * initial_sim +
                (1 - self.lambda_weight) * disc_sim
        )

        return final_similarity

    def retrieve(
            self,
            query_features: torch.Tensor,
            gallery_features: torch.Tensor,
            query_disc_features: Optional[torch.Tensor] = None,
            top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform retrieval.

        Args:
            query_features: Query features [Q, D]
            gallery_features: Gallery features [G, D]
            query_disc_features: Discriminative features [Q, D] (optional)
            top_k: Number of results to return

        Returns:
            similarity: Similarity matrix [Q, G]
            indices: Ranking indices [Q, G] or [Q, top_k]
        """
        if query_disc_features is not None:
            similarity = self.two_stage_retrieval(
                query_features, query_disc_features, gallery_features
            )
        else:
            similarity = self.compute_similarity(query_features, gallery_features)

        # Get rankings
        indices = similarity.argsort(dim=-1, descending=True)

        if top_k is not None:
            indices = indices[:, :top_k]

        return similarity, indices
