#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vision Encoder based on Vision Transformer.
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
    Residual attention block with attention weight extraction.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        ln_x = self.ln_1(x)
        attn_out, attn_weights = self.attn(
            ln_x, ln_x, ln_x,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.ln_2(x))

        return x, attn_weights


class VisionEncoder(nn.Module):
    """
    Vision Transformer encoder for CalibCLIP.

    Supports variable image sizes and stride for person ReID tasks.
    """

    def __init__(
            self,
            image_resolution: Tuple[int, int],  # (H, W)
            patch_size: int,
            stride_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int,
    ):
        super().__init__()

        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim

        # Calculate number of patches
        self.num_patches_h = (image_resolution[0] - patch_size) // stride_size + 1
        self.num_patches_w = (image_resolution[1] - patch_size) // stride_size + 1
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=stride_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        # Transformer blocks
        self.transformer = nn.Sequential(*[
            ResidualAttentionBlock(width, heads)
            for _ in range(layers)
        ])

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
            self,
            x: torch.Tensor,
            return_all_tokens: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            x: Input images of shape (B, 3, H, W)
            return_all_tokens: Whether to return all patch tokens

        Returns:
            cls_features: CLS token features (B, output_dim)
            patch_features: Patch token features (B, num_patches, width) if return_all_tokens
            info: Dictionary with intermediate information
        """
        batch_size = x.shape[0]

        # Patch embedding: (B, 3, H, W) -> (B, width, h, w)
        x = self.conv1(x)

        # Flatten and transpose: (B, width, h, w) -> (B, num_patches, width)
        x = x.reshape(batch_size, self.width, -1).permute(0, 2, 1)

        # Add CLS token
        cls_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add positional embedding
        x = x + self.positional_embedding.unsqueeze(0)

        x = self.ln_pre(x)

        # Pass through transformer (need to extract attention)
        attention_weights = []
        for block in self.transformer:
            x, attn = block(x, return_attention=True)
            if attn is not None:
                attention_weights.append(attn)

        x = self.ln_post(x)

        # Extract CLS and patch tokens
        cls_token_out = x[:, 0]
        patch_tokens = x[:, 1:]

        # Project CLS token
        cls_features = cls_token_out @ self.proj

        # Compute CLS attention from last layer
        if len(attention_weights) > 0:
            last_attn = attention_weights[-1]
            cls_attention = last_attn[:, :, 0, 1:].mean(dim=1)
        else:
            cls_attention = None

        info = {
            "attention_weights": attention_weights,
            "cls_attention": cls_attention,
            "cls_token": cls_token_out,
            "num_patches_h": self.num_patches_h,
            "num_patches_w": self.num_patches_w,
        }

        if return_all_tokens:
            return cls_features, patch_tokens, info
        else:
            return cls_features, None, info
