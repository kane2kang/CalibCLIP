#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Discriminative Concept Calibrator (DCC) module.

This module implements the textual feature calibration as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class DiscriminativeConceptCalibrator(nn.Module):
    """
    Discriminative Concept Calibrator (DCC).

    DCC differentiates between general and discriminative concepts within text queries.
    It mitigates challenges posed by generic concepts and improves representations
    of discriminative concepts to strengthen differentiation among similar samples.
    """

    def __init__(
            self,
            embed_dim: int,
            lambda_weight: float = 0.5,
            temperature: float = 0.01,
    ):
        """
        Args:
            embed_dim: Dimension of text embeddings
            lambda_weight: Weight for combining global and discriminative features
            temperature: Temperature for softmax in attention computation
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.lambda_weight = lambda_weight
        self.temperature = temperature

    def compute_concept_frequency(
            self,
            token_ids: torch.Tensor,
            corpus_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute token frequency as proxy for concept generality.
        Higher frequency tokens are more likely to be general concepts.

        Args:
            token_ids: Token IDs for current batch (B, seq_len)
            corpus_token_ids: Token IDs for entire corpus (N, seq_len), optional

        Returns:
            frequency_scores: Normalized frequency scores (B, seq_len)
        """
        if corpus_token_ids is not None:
            # Compute frequency from corpus
            flat_corpus = corpus_token_ids.flatten()
            unique_tokens, counts = torch.unique(flat_corpus, return_counts=True)

            # Create frequency lookup
            max_token = int(unique_tokens.max().item()) + 1
            freq_lookup = torch.zeros(max_token, device=token_ids.device)
            freq_lookup[unique_tokens] = counts.float()

            # Lookup frequencies for current tokens
            frequency_scores = freq_lookup[token_ids.clamp(max=max_token - 1)]
        else:
            # Use batch frequency as approximation
            flat_tokens = token_ids.flatten()
            unique_tokens, counts = torch.unique(flat_tokens, return_counts=True)

            max_token = int(unique_tokens.max().item()) + 1
            freq_lookup = torch.zeros(max_token, device=token_ids.device)
            freq_lookup[unique_tokens] = counts.float()

            frequency_scores = freq_lookup[token_ids.clamp(max=max_token - 1)]

        # Normalize to [0, 1]
        frequency_scores = frequency_scores / (frequency_scores.max() + 1e-8)

        return frequency_scores

    def identify_discriminative_tokens(
            self,
            eot_attention: torch.Tensor,
            frequency_scores: torch.Tensor,
            token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify discriminative vs general tokens.

        Discriminative tokens have:
        - High attention from EOT token (semantically important)
        - Low frequency in corpus (distinctive)

        Args:
            eot_attention: EOT attention to all tokens (B, heads, seq_len)
            frequency_scores: Token frequency scores (B, seq_len)
            token_mask: Valid token mask (B, seq_len)

        Returns:
            discriminative_weights: Weights for discriminative tokens (B, seq_len)
            general_weights: Weights for general tokens (B, seq_len)
        """
        # Average attention across heads
        if eot_attention.dim() == 3:
            avg_attention = eot_attention.mean(dim=1)  # (B, seq_len)
        else:
            avg_attention = eot_attention

        # Discriminativeness = high attention * low frequency
        # Invert frequency so low frequency = high score
        inverse_frequency = 1.0 - frequency_scores

        # Compute discriminative score
        discriminative_score = avg_attention * inverse_frequency

        # Apply token mask if provided
        if token_mask is not None:
            discriminative_score = discriminative_score * token_mask

        # Normalize to get weights
        discriminative_weights = F.softmax(
            discriminative_score / self.temperature, dim=-1
        )

        # General weights are inverse
        general_score = avg_attention * frequency_scores
        if token_mask is not None:
            general_score = general_score * token_mask
        general_weights = F.softmax(general_score / self.temperature, dim=-1)

        return discriminative_weights, general_weights

    def aggregate_weighted_features(
            self,
            token_features: torch.Tensor,
            weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate token features using weights.

        Args:
            token_features: Token embeddings (B, seq_len, embed_dim)
            weights: Aggregation weights (B, seq_len)

        Returns:
            aggregated: Weighted aggregated features (B, embed_dim)
        """
        # Weighted sum
        aggregated = torch.einsum('bn,bnd->bd', weights, token_features)
        return aggregated

    def forward(
            self,
            eot_features: torch.Tensor,
            token_features: torch.Tensor,
            eot_attention: torch.Tensor,
            token_ids: Optional[torch.Tensor] = None,
            token_mask: Optional[torch.Tensor] = None,
            corpus_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply Discriminative Concept Calibration.

        Args:
            eot_features: EOT token features (B, embed_dim)
            token_features: All token features (B, seq_len, embed_dim)
            eot_attention: EOT attention weights (B, heads, seq_len) or (B, seq_len)
            token_ids: Token IDs for frequency computation (B, seq_len)
            token_mask: Valid token mask (B, seq_len)
            corpus_token_ids: Corpus token IDs for frequency (N, seq_len)

        Returns:
            calibrated_features: Calibrated text features (B, embed_dim)
            info: Dictionary with intermediate results
        """
        batch_size = eot_features.shape[0]

        # Compute frequency scores if token_ids provided
        if token_ids is not None:
            frequency_scores = self.compute_concept_frequency(
                token_ids, corpus_token_ids
            )
        else:
            # Use uniform frequency if not provided
            frequency_scores = torch.ones(
                token_features.shape[:2], device=token_features.device
            ) * 0.5

        # Identify discriminative and general tokens
        discriminative_weights, general_weights = self.identify_discriminative_tokens(
            eot_attention, frequency_scores, token_mask
        )

        # Aggregate features
        discriminative_features = self.aggregate_weighted_features(
            token_features, discriminative_weights
        )
        general_features = self.aggregate_weighted_features(
            token_features, general_weights
        )

        # Calibrated features: emphasize discriminative, de-emphasize general
        # f_calibrated = f_eot + λ * (f_disc - f_gen)
        calibrated_features = eot_features + self.lambda_weight * (
                discriminative_features - general_features
        )

        # Normalize
        calibrated_features = F.normalize(calibrated_features, p=2, dim=-1)

        info = {
            "discriminative_weights": discriminative_weights,
            "general_weights": general_weights,
            "frequency_scores": frequency_scores,
            "discriminative_features": discriminative_features,
            "general_features": general_features,
        }

        return calibrated_features, info
