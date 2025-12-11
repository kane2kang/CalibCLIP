#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Contrastive Visual Enhancer (CVE) module.

This module implements the visual feature calibration as described in the paper:
"CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ContrastiveVisualEnhancer(nn.Module):
    """
    Contrastive Visual Enhancer (CVE).

    CVE decouples visual features into target and low information regions,
    identifies dominant tokens, and dynamically suppresses their representations
    to enhance discriminative visual features.

    The key insight is that the CLS token may be dominated by a few patches
    with high attention weights, which can suppress other discriminative features.
    CVE addresses this by:
    1. Identifying dominant patches via attention threshold
    2. Extracting contextual information from non-dominant patches
    3. Enhancing CLS representation with filtered contextual features
    """

    def __init__(
            self,
            embed_dim: int,
            output_dim: int,
            residual_coefficient: float = 0.1,
            attention_percentile: float = 90.0,
    ):
        """
        Args:
            embed_dim: Dimension of input patch embeddings
            output_dim: Dimension of output features
            residual_coefficient: Coefficient for residual connection (default: 0.1)
            attention_percentile: Percentile threshold for identifying dominant patches
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.residual_coefficient = residual_coefficient
        self.attention_percentile = attention_percentile

        # Projection layers
        self.context_proj = nn.Linear(embed_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )

    def compute_attention_threshold(
            self,
            cls_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive attention threshold based on percentile.

        Args:
            cls_attention: CLS token attention weights (B, num_patches)

        Returns:
            threshold: Per-sample attention threshold (B, 1)
        """
        # Use percentile-based threshold
        threshold = torch.quantile(
            cls_attention,
            self.attention_percentile / 100.0,
            dim=-1,
            keepdim=True
        )
        return threshold

    def identify_dominant_patches(
            self,
            cls_attention: torch.Tensor,
            threshold: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify dominant and non-dominant patches based on attention.

        Args:
            cls_attention: CLS attention weights (B, num_patches)
            threshold: Attention threshold (B, 1)

        Returns:
            dominant_mask: Binary mask for dominant patches (B, num_patches)
            context_mask: Binary mask for context patches (B, num_patches)
        """
        # Dominant patches have attention above threshold
        dominant_mask = (cls_attention >= threshold).float()

        # Context patches are non-dominant
        context_mask = 1.0 - dominant_mask

        return dominant_mask, context_mask

    def aggregate_context_features(
            self,
            patch_tokens: torch.Tensor,
            context_mask: torch.Tensor,
            cls_attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate context features from non-dominant patches.

        Args:
            patch_tokens: Patch token embeddings (B, num_patches, embed_dim)
            context_mask: Binary mask for context patches (B, num_patches)
            cls_attention: CLS attention weights for weighting (B, num_patches)

        Returns:
            context_features: Aggregated context features (B, embed_dim)
        """
        # Weight context patches by their attention (normalized within context)
        context_attention = cls_attention * context_mask
        context_attention_sum = context_attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized_weights = context_attention / context_attention_sum

        # Weighted aggregation
        context_features = torch.einsum(
            'bn,bnd->bd',
            normalized_weights,
            patch_tokens
        )

        return context_features

    def forward(
            self,
            cls_token: torch.Tensor,
            patch_tokens: torch.Tensor,
            cls_attention: torch.Tensor,
            cls_projected: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply Contrastive Visual Enhancement.

        Args:
            cls_token: CLS token embedding (B, embed_dim)
            patch_tokens: Patch token embeddings (B, num_patches, embed_dim)
            cls_attention: CLS attention to patches (B, num_patches)
            cls_projected: Pre-projected CLS features (B, output_dim), optional

        Returns:
            enhanced_features: Enhanced visual features (B, output_dim)
            info: Dictionary containing intermediate results
        """
        batch_size = cls_token.shape[0]

        # Compute adaptive threshold
        threshold = self.compute_attention_threshold(cls_attention)

        # Identify dominant and context patches
        dominant_mask, context_mask = self.identify_dominant_patches(
            cls_attention, threshold
        )

        # Aggregate context
        # Aggregate context features
        context_features = self.aggregate_context_features(
            patch_tokens, context_mask, cls_attention
        )

        # Project context features
        context_projected = self.context_proj(context_features)

        # If cls_projected not provided, project cls_token
        if cls_projected is None:
            cls_projected = self.context_proj(cls_token)

        # Gated fusion
        combined = torch.cat([cls_projected, context_projected], dim=-1)
        gate_weights = self.gate(combined)

        # Enhanced features with residual connection
        enhanced_features = cls_projected + self.residual_coefficient * gate_weights * context_projected

        # Normalize
        enhanced_features = F.normalize(enhanced_features, p=2, dim=-1)

        info = {
            "threshold": threshold,
            "dominant_mask": dominant_mask,
            "context_mask": context_mask,
            "context_features": context_projected,
            "gate_weights": gate_weights,
            "num_dominant": dominant_mask.sum(dim=-1).mean().item(),
            "num_context": context_mask.sum(dim=-1).mean().item(),
        }

        return enhanced_features, info

