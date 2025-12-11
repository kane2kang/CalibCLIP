#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-head attention module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
            batch_first: bool = True,  # 添加 batch_first 参数
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first  # 保存参数

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim) if batch_first=True
               or (seq_len, batch_size, embed_dim) if batch_first=False
            attention_mask: Optional mask of shape (seq_len, seq_len) for causal mask
                           or (batch_size, seq_len) for padding mask
            return_attention: Whether to return attention weights

        Returns:
            output: Output tensor of same shape as input
            attention_weights: Optional attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        # Handle batch_first
        if not self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, dim) -> (batch, seq_len, dim)

        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.shape[0] == seq_len and attention_mask.shape[1] == seq_len:
                    # Causal mask (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                else:
                    # Padding mask (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1)

            # For masks with -inf values (causal mask), add directly
            # For masks with 0/1 values (padding mask), use masked_fill
            if attention_mask.dtype == torch.bool:
                attention_scores = attention_scores.masked_fill(
                    attention_mask == 0, float('-inf')
                )
            else:
                attention_scores = attention_scores + attention_mask

        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)

        # Handle batch_first for output
        if not self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq_len, dim) -> (seq_len, batch, dim)

        if return_attention:
            return output, attention_weights
        return output, None
