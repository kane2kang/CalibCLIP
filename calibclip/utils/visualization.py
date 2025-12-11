#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities.
"""

import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 兼容不同版本的 PIL
try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BILINEAR = Image.BILINEAR


def visualize_attention(
        image: Union[np.ndarray, torch.Tensor],
        attention: Union[np.ndarray, torch.Tensor],
        num_patches: Tuple[int, int],  # 修改: 直接传入 patch 数量 (num_y, num_x)
        save_path: Optional[str] = None,
        title: str = "Attention Map",
        alpha: float = 0.5,
) -> plt.Figure:
    """
    Visualize attention map overlaid on image.

    Args:
        image: Original image [H, W, 3] or [3, H, W]
        attention: Attention weights [num_patches] or [num_y, num_x]
        num_patches: Number of patches (num_y, num_x)
        save_path: Path to save figure
        title: Figure title
        alpha: Transparency of attention overlay

    Returns:
        fig: Matplotlib figure
    """
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
            image = image.transpose(1, 2, 0)

    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()

    # Denormalize image if needed (assuming normalized with CLIP stats)
    if image.max() <= 1.0:
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image = image * std + mean
        image = np.clip(image, 0, 1)

    H, W = image.shape[:2]
    num_y, num_x = num_patches

    # Reshape attention to spatial dimensions
    attn_map = attention.reshape(num_y, num_x)

    # Resize attention map to image size
    attn_map_pil = Image.fromarray(attn_map.astype(np.float32))
    attn_map = np.array(attn_map_pil.resize((W, H), BILINEAR))

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention map
    im = axes[1].imshow(attn_map, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attn_map, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_retrieval(
        query_text: str,
        gallery_images: List[Union[np.ndarray, str]],
        rankings: List[int],
        pids: List[int],
        query_pid: int,
        top_k: int = 10,
        save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize retrieval results.

    Args:
        query_text: Query text
        gallery_images: List of gallery images (arrays or paths)
        rankings: Ranking indices
        pids: Person IDs for gallery images
        query_pid: Query person ID
        top_k: Number of top results to show
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    # Get top-k results
    top_indices = rankings[:top_k]

    # Create figure
    ncols = min(5, top_k)
    nrows = (top_k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))

    # Flatten axes for easier indexing
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, idx in enumerate(top_indices):
        if i >= len(axes):
            break

        ax = axes[i]

        # Load image if path
        img = gallery_images[idx]
        if isinstance(img, str):
            img = np.array(Image.open(img).convert("RGB"))

        pid = pids[idx]
        is_correct = (pid == query_pid)

        ax.imshow(img)
        ax.set_title(f"Rank {i + 1}\nPID: {pid}",
                     color="green" if is_correct else "red")
        ax.axis("off")

        # Add border
        border_color = "green" if is_correct else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Hide unused axes
    for ax in axes[len(top_indices):]:
        ax.axis("off")
        ax.set_visible(False)

    # Add query text (truncate if too long)
    display_text = query_text if len(query_text) < 100 else query_text[:97] + "..."
    fig.suptitle(f"Query: {display_text}\nQuery PID: {query_pid}",
                 fontsize=10, wrap=True)

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_cve_analysis(
        image: Union[np.ndarray, torch.Tensor],
        cls_attention: Union[np.ndarray, torch.Tensor],
        dominant_mask: Union[np.ndarray, torch.Tensor],
        context_mask: Union[np.ndarray, torch.Tensor],
        gate_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_patches: Tuple[int, int] = None,
        save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize CVE analysis results.

    Args:
        image: Original image [H, W, 3] or [3, H, W]
        cls_attention: CLS attention weights [N]
        dominant_mask: Dominant patch mask [N]
        context_mask: Context patch mask [N]
        gate_weights: Gate weights (scalar or [D])
        num_patches: Number of patches (num_y, num_x)
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """

    # Convert to numpy
    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    image = to_numpy(image)
    cls_attention = to_numpy(cls_attention)
    dominant_mask = to_numpy(dominant_mask)
    context_mask = to_numpy(context_mask)
    gate_weights = to_numpy(gate_weights)

    # Handle image format
    if image.shape[0] == 3:  # [3, H, W] -> [H, W, 3]
        image = image.transpose(1, 2, 0)

    # Denormalize if needed
    if image.max() <= 1.0:
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        image = image * std + mean
        image = np.clip(image, 0, 1)

    H, W = image.shape[:2]

    # Infer num_patches if not provided
    if num_patches is None:
        num_total = len(cls_attention)
        # Assume roughly 3:1 aspect ratio for person images
        num_y = int(np.sqrt(num_total * 3))
        num_x = num_total // num_y
        num_patches = (num_y, num_x)

    num_y, num_x = num_patches

    def reshape_and_resize(x):
        if x is None:
            return None
        x = x.flatten()[:num_y * num_x].reshape(num_y, num_x)
        x_pil = Image.fromarray(x.astype(np.float32))
        return np.array(x_pil.resize((W, H), BILINEAR))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # CLS attention
    attn_map = reshape_and_resize(cls_attention)
    im1 = axes[0, 1].imshow(attn_map, cmap="jet")
    axes[0, 1].set_title("CLS Attention")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Dominant mask
    dom_map = reshape_and_resize(dominant_mask)
    axes[0, 2].imshow(dom_map, cmap="Reds")
    axes[0, 2].set_title(f"Dominant Mask (n={dominant_mask.sum():.0f})")
    axes[0, 2].axis("off")

    # Context mask
    ctx_map = reshape_and_resize(context_mask)
    axes[1, 0].imshow(ctx_map, cmap="Blues")
    axes[1, 0].set_title(f"Context Mask (n={context_mask.sum():.0f})")
    axes[1, 0].axis("off")

    # Context attention (attention weighted by context mask)
    context_attention = cls_attention * context_mask
    ctx_attn_map = reshape_and_resize(context_attention)
    axes[1, 1].imshow(ctx_attn_map, cmap="hot")
    axes[1, 1].set_title("Context Attention")
    axes[1, 1].axis("off")

    # Overlay
    axes[1, 2].imshow(image)
    # Show dominant in red, context in blue
    if dom_map is not None:
        dom_overlay = np.zeros((H, W, 4))
        dom_overlay[:, :, 0] = dom_map / (dom_map.max() + 1e-8)  # Red channel
        dom_overlay[:, :, 3] = dom_map / (dom_map.max() + 1e-8) * 0.5  # Alpha
        axes[1, 2].imshow(dom_overlay)
    axes[1, 2].set_title("Dominant (Red) Overlay")
    axes[1, 2].axis("off")

    # Add gate weight info if available
    if gate_weights is not None:
        gate_mean = gate_weights.mean() if gate_weights.ndim > 0 else gate_weights
        fig.suptitle(f"CVE Analysis (Gate Weight Mean: {gate_mean:.4f})")
    else:
        fig.suptitle("CVE Analysis")

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
