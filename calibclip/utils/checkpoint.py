#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint utilities.
"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
        state: Dict[str, Any],
        save_path: str,
        is_best: bool = False,
):
    """
    Save checkpoint.

    Args:
        state: State dictionary to save
        save_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)

    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), "best.pth")
        torch.save(state, best_path)


def load_checkpoint(
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
        strict: bool = True,
) -> Dict[str, Any]:
    """
    Load checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into (optional)
        device: Device to load checkpoint
        strict: Whether to strictly enforce matching keys

    Returns:
        checkpoint: Loaded checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None:
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    return checkpoint
