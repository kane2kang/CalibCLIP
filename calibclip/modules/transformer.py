#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformer architectures for vision and text encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from collections import OrderedDict

from .layers import LayerNorm, QuickGELU
from .attention import MultiHeadAttention


class ResidualAttentionBlock(nn.Module):
    """
    Transformer block with residual connections.
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            attn_mask: torch.Tensor = None,
            dropout: float = 0.0,
    ):
        super().__init__()

        # 使用 batch_first=True，输入输出格式为 [B, L, D]
        self.attn = MultiHeadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [L, B, D] (seq_first format for compatibility)
            return_attention: Whether to return attention weights

        Returns:
            output: Output tensor [L, B, D]
            attn_weights: Attention weights [B, H, L, L] if return_attention
        """
        # Convert to batch_first: [L, B, D] -> [B, L, D]
        x_bf = x.transpose(0, 1)

        # Prepare attention mask
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)

        # Self-attention
        ln_x = self.ln_1(x_bf)
        attn_out, attn_weights = self.attn(ln_x, attention_mask=attn_mask, return_attention=return_attention)
        x_bf = x_bf + attn_out

        # MLP
        x_bf = x_bf + self.mlp(self.ln_2(x_bf))

        # Convert back: [B, L, D] -> [L, B, D]
        x = x_bf.transpose(0, 1)

        return x, attn_weights


class Transformer(nn.Module):
    """
    Standard Transformer encoder.
    """

    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            attn_mask: torch.Tensor = None,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask, dropout)
            for _ in range(layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False,
            return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[List]]:
        """
        Forward pass through all transformer layers.

        Args:
            x: Input tensor [L, B, D]
            return_attention: Whether to return attention weights from each layer
            return_hidden_states: Whether to return hidden states from each layer

        Returns:
            output: Final output tensor [L, B, D]
            attn_weights_list: List of attention weights from each layer
            hidden_states_list: List of hidden states from each layer
        """
        attn_weights_list = [] if return_attention else None
        hidden_states_list = [] if return_hidden_states else None

        for block in self.resblocks:
            if return_hidden_states:
                hidden_states_list.append(x.clone())

            x, attn_weights = block(x, return_attention=return_attention)

            if return_attention and attn_weights is not None:
                attn_weights_list.append(attn_weights)

        if return_hidden_states:
            hidden_states_list.append(x.clone())

        return x, attn_weights_list, hidden_states_list


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image encoding.
    """

    def __init__(
            self,
            input_resolution: Tuple[int, int],
            patch_size: int,
            stride_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.output_dim = output_dim

        # Calculate number of patches
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_patches = self.num_y * self.num_x

        # Patch embedding
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=stride_size,
            bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
            self,
            x: torch.Tensor,
            return_all_features: bool = False,
            return_attention: bool = False,
    ) -> Tuple:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            return_all_features: Whether to return all patch features
            return_attention: Whether to return attention weights

        Returns:
            cls_features: CLS token features [B, D]
            patch_features: Patch features [B, N, D] if return_all_features
            attn_weights: Attention weights if return_attention
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.conv1(x)  # [B, width, num_y, num_x]
        x = x.reshape(batch_size, x.shape[1], -1)  # [B, width, num_patches]
        x = x.permute(0, 2, 1)  # [B, num_patches, width]

        # Add CLS token
        cls_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1 + num_patches, width]

        # Add positional embedding
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        # Transformer (expects [L, B, D])
        x = x.permute(1, 0, 2)  # [L, B, D]
        x, attn_list, _ = self.transformer(
            x,
            return_attention=return_attention,
            return_hidden_states=False
        )
        x = x.permute(1, 0, 2)  # [B, L, D]

        # Post-processing
        x = self.ln_post(x)

        cls_features = x[:, 0] @ self.proj  # [B, output_dim]

        if return_all_features:
            patch_features = x[:, 1:]  # [B, num_patches, width] (不投影patch)
            if return_attention:
                return cls_features, patch_features, attn_list
            return cls_features, patch_features, None

        if return_attention:
            return cls_features, None, attn_list
        return cls_features, None, None


class TextTransformer(nn.Module):
    """
    Text Transformer for text encoding.
    """

    def __init__(
            self,
            context_length: int,
            vocab_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))

        # Causal attention mask
        self.register_buffer(
            "attn_mask",
            self.build_attention_mask(context_length),
            persistent=False
        )

        self.transformer = Transformer(width, layers, heads, self.attn_mask)

        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.width ** -0.5)

    @staticmethod
    def build_attention_mask(context_length: int) -> torch.Tensor:
        """Build causal attention mask."""
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(
            self,
            text: torch.Tensor,
            return_all_features: bool = False,
            return_attention: bool = False,
    ) -> Tuple:
        """
        Forward pass.

        Args:
            text: Tokenized text [B, L]
            return_all_features: Whether to return all token features
            return_attention: Whether to return attention weights

        Returns:
            eot_features: EOT token features [B, D]
            token_features: All token features [B, L, width] if return_all_features
            attn_weights: Attention weights if return_attention
            eot_indices: EOT token indices [B]
        """
        x = self.token_embedding(text)  # [B, L, width]
        x = x + self.positional_embedding[:x.shape[1]]

        x = x.permute(1, 0, 2)  # [L, B, D]
        x, attn_list, _ = self.transformer(
            x,
            return_attention=return_attention,
            return_hidden_states=False
        )
        x = x.permute(1, 0, 2)  # [B, L, D]

        x = self.ln_final(x)

        # Get EOT token position (last non-padded token)
        eot_indices = text.argmax(dim=-1)  # [B]
        eot_features = x[torch.arange(x.shape[0], device=x.device), eot_indices]  # [B, width]
        eot_features = eot_features @ self.text_projection  # [B, output_dim]

        if return_all_features:
            token_features = x  # [B, L, width] (不投影所有token)
            if return_attention:
                return eot_features, token_features, attn_list, eot_indices
            return eot_features, token_features, None, eot_indices

        if return_attention:
            return eot_features, None, attn_list, eot_indices
        return eot_features, None, None, eot_indices
