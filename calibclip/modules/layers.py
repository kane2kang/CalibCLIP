#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic layers and components.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Fast GELU approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Dropout(nn.Module):
    """Dropout layer with optional inplace."""

    def __init__(self, p: float = 0.0, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p > 0 and self.training:
            return nn.functional.dropout(x, p=self.p, training=True, inplace=self.inplace)
        return x


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            act_layer: nn.Module = QuickGELU,
            drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
