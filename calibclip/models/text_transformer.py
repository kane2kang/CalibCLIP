#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Encoder based on Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Quick GELU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Residual attention block for text encoder.
    """

    def __init__(self, d_model: int, n_head: int, causal: bool = True):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.causal = causal

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        ln_x = self.ln_1(x)
        attn_out, attn_weights = self.attn(
            ln_x, ln_x, ln_x,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.ln_2(x))

        return x, attn_weights


class TextEncoder(nn.Module):
    """
    Text Transformer encoder for CalibCLIP.
    """

    def __init__(
            self,
            context_length: int,
            vocab_size: int,
            width: int,
            heads: int,
            layers: int,
            output_dim: int,
    ):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.layers = layers
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.randn(context_length, width))

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, causal=True)
            for _ in range(layers)
        ])

        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.randn(width, output_dim))

    def build_attention_mask(self, context_length: int) -> torch.Tensor:
        """Build causal attention mask."""
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
            self,
            text: torch.Tensor,
            return_all_tokens: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            text: Input token ids of shape (B, context_length)
            return_all_tokens: Whether to return all token features

        Returns:
            eot_features: EOT token features (B, output_dim)
            token_features: All token features (B, context_length, width) if return_all_tokens
            info: Dictionary with intermediate information
        """
        batch_size = text.shape[0]

        # Token embedding
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:text.shape[1]]

        # Build attention mask
        attn_mask = self.build_attention_mask(text.shape[1]).to(x.device)

        # Pass through transformer
        attention_weights = []
        for block in self.transformer:
            x, attn = block(x, attn_mask=attn_mask, return_attention=True)
            if attn is not None:
                attention_weights.append(attn)

        x = self.ln_final(x)

        # Find EOT token positions (highest value in each sequence)
        eot_indices = text.argmax(dim=-1)

        # Extract EOT features
        eot_features = x[torch.arange(batch_size, device=x.device), eot_indices]
        eot_features = eot_features @ self.text_projection

        # Get EOT attention from last layer
        if len(attention_weights) > 0:
            last_attn = attention_weights[-1]
            eot_attention = last_attn[
                torch.arange(batch_size, device=x.device), :, eot_indices
            ]
        else:
            eot_attention = None

        info = {
            "attention_weights": attention_weights,
            "eot_attention": eot_attention,
            "eot_indices": eot_indices,
        }

        if return_all_tokens:
            return eot_features, x, info
        else:
            return eot_features, None, info
