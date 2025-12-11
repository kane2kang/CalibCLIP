#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CalibCLIP main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
import clip
import math

from .cve import ContrastiveVisualEnhancer
from .dcc import DiscriminativeConceptCalibrator


class CalibCLIP(nn.Module):
    """
    CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval.

    This is a training-free method that calibrates CLIP's visual and textual features
    to improve text-driven image retrieval performance.
    """

    def __init__(
            self,
            clip_model: nn.Module,
            image_size: Tuple[int, int] = (384, 128),
            stride_size: int = 16,
            cve_config: Optional[Dict] = None,
            dcc_config: Optional[Dict] = None,
    ):
        """
        Args:
            clip_model: Pre-trained CLIP model
            image_size: Input image size (H, W)
            stride_size: Stride for patch embedding
            cve_config: Configuration for CVE module
            dcc_config: Configuration for DCC module
        """
        super().__init__()

        self.clip_model = clip_model
        self.image_size = image_size
        self.stride_size = stride_size

        # Get CLIP dimensions
        self.embed_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, 'output_dim') else 512
        self.vision_width = clip_model.visual.conv1.out_channels
        self.patch_size = clip_model.visual.conv1.kernel_size[0]

        # Modify visual encoder for custom image size and stride
        self._modify_visual_encoder()

        # Initialize CVE
        cve_config = cve_config or {}
        self.cve = ContrastiveVisualEnhancer(
            embed_dim=self.vision_width,
            output_dim=self.embed_dim,
            residual_coefficient=cve_config.get('residual_coefficient', 0.1),
            attention_percentile=cve_config.get('attention_percentile', 90.0),
        )

        # Initialize DCC
        dcc_config = dcc_config or {}
        self.dcc = DiscriminativeConceptCalibrator(
            embed_dim=self.embed_dim,
            lambda_weight=dcc_config.get('lambda_weight', 0.5),
            temperature=dcc_config.get('temperature', 0.01),
        )

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def _modify_visual_encoder(self):
        """Modify CLIP's visual encoder for custom image size and stride."""
        visual = self.clip_model.visual

        # Calculate new number of patches
        h_patches = (self.image_size[0] - self.patch_size) // self.stride_size + 1
        w_patches = (self.image_size[1] - self.patch_size) // self.stride_size + 1
        self.num_patches = h_patches * w_patches
        self.h_patches = h_patches
        self.w_patches = w_patches

        # Modify conv1 stride if needed
        if self.stride_size != self.patch_size:
            visual.conv1.stride = (self.stride_size, self.stride_size)

        # Interpolate positional embeddings
        old_pos_embed = visual.positional_embedding.data
        old_num_patches = old_pos_embed.shape[0] - 1  # Exclude CLS token
        old_size = int(old_num_patches ** 0.5)

        if self.num_patches != old_num_patches:
            # Separate CLS and patch embeddings
            cls_embed = old_pos_embed[0:1]
            patch_embed = old_pos_embed[1:]

            # Reshape to 2D grid
            patch_embed = patch_embed.reshape(old_size, old_size, -1)
            patch_embed = patch_embed.permute(2, 0, 1).unsqueeze(0)  # (1, dim, h, w)

            # Interpolate to new size
            new_patch_embed = F.interpolate(
                patch_embed.float(),
                size=(h_patches, w_patches),
                mode='bicubic',
                align_corners=False,
            )
            new_patch_embed = new_patch_embed.squeeze(0).permute(1, 2, 0)  # (h, w, dim)
            new_patch_embed = new_patch_embed.reshape(-1, old_pos_embed.shape[-1])

            # Combine CLS and new patch embeddings
            new_pos_embed = torch.cat([cls_embed, new_patch_embed], dim=0)
            visual.positional_embedding = nn.Parameter(new_pos_embed.to(old_pos_embed.dtype))

    def encode_image(
            self,
            images: torch.Tensor,
            apply_cve: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode images with optional CVE enhancement.

        Args:
            images: Input images (B, 3, H, W)
            apply_cve: Whether to apply CVE enhancement

        Returns:
            image_features: Encoded image features (B, embed_dim)
            info: Dictionary with intermediate results
        """
        visual = self.clip_model.visual

        # Patch embedding
        x = visual.conv1(images)  # (B, width, h, w)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, width, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, width)

        # Add CLS token
        cls_token = visual.class_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # (B, 1+num_patches, width)

        # Add positional embedding
        x = x + visual.positional_embedding

        # Pre-normalization
        x = visual.ln_pre(x)

        # Transformer with attention extraction
        x = x.permute(1, 0, 2)  # (seq, B, width) for transformer

        attention_weights = []
        for block in visual.transformer.resblocks:
            # Extract attention from each block
            x_norm = block.ln_1(x)
            attn_out, attn_weight = block.attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + attn_out
            x = x + block.mlp(block.ln_2(x))
            attention_weights.append(attn_weight)

        x = x.permute(1, 0, 2)  # (B, seq, width)
        x = visual.ln_post(x)

        # Extract CLS and patch tokens
        cls_token_out = x[:, 0]  # (B, width)
        patch_tokens = x[:, 1:]  # (B, num_patches, width)

        # Project CLS token
        if visual.proj is not None:
            cls_features = cls_token_out @ visual.proj  # (B, embed_dim)
        else:
            cls_features = cls_token_out

        # Get CLS attention from last layer
        last_attn = attention_weights[-1]  # (B, heads, seq, seq)
        cls_attention = last_attn[:, :, 0, 1:].mean(dim=1)  # (B, num_patches)

        info = {
            "cls_token": cls_token_out,
            "patch_tokens": patch_tokens,
            "cls_attention": cls_attention,
            "attention_weights": attention_weights,
            "cls_features_original": cls_features.clone(),
        }

        # Apply CVE if enabled
        if apply_cve:
            enhanced_features, cve_info = self.cve(
                cls_token=cls_token_out,
                patch_tokens=patch_tokens,
                cls_attention=cls_attention,
                cls_projected=cls_features,
            )
            info["cve_info"] = cve_info
            image_features = enhanced_features
        else:
            image_features = F.normalize(cls_features, p=2, dim=-1)

        return image_features, info

    def encode_text(
            self,
            text: torch.Tensor,
            apply_dcc: bool = True,
            corpus_token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode text with optional DCC calibration.

        Args:
            text: Input token IDs (B, context_length)
            apply_dcc: Whether to apply DCC calibration
            corpus_token_ids: Token IDs for corpus frequency computation

        Returns:
            text_features: Encoded text features (B, embed_dim)
            info: Dictionary with intermediate results
        """
        # Token embedding
        x = self.clip_model.token_embedding(text)  # (B, seq, width)
        x = x + self.clip_model.positional_embedding[:text.shape[1]]

        # Transformer with attention extraction
        x = x.permute(1, 0, 2)  # (seq, B, width)

        # Build causal mask
        seq_len = text.shape[1]
        attn_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=text.device) * float('-inf'),
            diagonal=1
        )

        attention_weights = []
        for block in self.clip_model.transformer.resblocks:
            x_norm = block.ln_1(x)
            attn_out, attn_weight = block.attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            x = x + attn_out
            x = x + block.mlp(block.ln_2(x))
            attention_weights.append(attn_weight)

        x = x.permute(1, 0, 2)  # (B, seq, width)
        x = self.clip_model.ln_final(x)

        # Find EOT positions
        eot_indices = text.argmax(dim=-1)  # (B,)

        # Extract EOT features
        eot_features = x[torch.arange(x.shape[0]), eot_indices]  # (B, width)

        # Project
        if hasattr(self.clip_model, 'text_projection') and self.clip_model.text_projection is not None:
            eot_features = eot_features @ self.clip_model.text_projection  # (B, embed_dim)

        # Get EOT attention from last layer
        last_attn = attention_weights[-1]  # (B, heads, seq, seq)
        eot_attention = last_attn[
                        torch.arange(last_attn.shape[0]), :, eot_indices
                        ]  # (B, heads, seq)

        # Create token mask (valid tokens before EOT)
        token_mask = torch.arange(seq_len, device=text.device).unsqueeze(0) <= eot_indices.unsqueeze(1)
        token_mask = token_mask.float()

        info = {
            "token_features": x,
            "eot_features_original": eot_features.clone(),
            "eot_attention": eot_attention,
            "eot_indices": eot_indices,
            "token_mask": token_mask,
            "attention_weights": attention_weights,
        }

        # Apply DCC if enabled
        if apply_dcc:
            calibrated_features, dcc_info = self.dcc(
                eot_features=eot_features,
                token_features=x,
                eot_attention=eot_attention,
                token_ids=text,
                token_mask=token_mask,
                corpus_token_ids=corpus_token_ids,
            )
            info["dcc_info"] = dcc_info
            text_features = calibrated_features
        else:
            text_features = F.normalize(eot_features, p=2, dim=-1)

        return text_features, info

    def forward(
            self,
            images: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            apply_cve: bool = True,
            apply_dcc: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CalibCLIP.

        Args:
            images: Input images (B, 3, H, W)
            text: Input token IDs (B, context_length)
            apply_cve: Whether to apply CVE
            apply_dcc: Whether to apply DCC

        Returns:
            Dictionary containing image and/or text features
        """
        outputs = {}

        if images is not None:
            image_features, image_info = self.encode_image(images, apply_cve=apply_cve)
            outputs["image_features"] = image_features
            outputs["image_info"] = image_info

        if text is not None:
            text_features, text_info = self.encode_text(text, apply_dcc=apply_dcc)
            outputs["text_features"] = text_features
            outputs["text_info"] = text_info

        return outputs


def build_calibclip(
        model_name: str = "ViT-B/16",
        image_size: Tuple[int, int] = (384, 128),
        stride_size: int = 16,
        device: Union[str, torch.device] = "cuda",
        cve_config: Optional[Dict] = None,
        dcc_config: Optional[Dict] = None,
) -> CalibCLIP:
    """
    Build CalibCLIP model.

    Args:
        model_name: CLIP model name
        image_size: Input image size (H, W)
        stride_size: Stride for patch embedding
        device: Device to load model
        cve_config: CVE configuration
        dcc_config: DCC configuration

    Returns:
        CalibCLIP model
    """
    # Load CLIP model
    clip_model, _ = clip.load(model_name, device=device)

    # Build CalibCLIP
    model = CalibCLIP(
        clip_model=clip_model,
        image_size=image_size,
        stride_size=stride_size,
        cve_config=cve_config,
        dcc_config=dcc_config,
    )

    return model.to(device)


def load_pretrained_weights(
        model: CalibCLIP,
        pretrained_path: str,
        device: str = "cpu",
) -> CalibCLIP:
    """
    Load pretrained weights into CalibCLIP.

    Note: CalibCLIP is training-free and uses CLIP weights directly.
    This function is mainly for loading fine-tuned CVE/DCC weights if any.

    Args:
        model: CalibCLIP model
        pretrained_path: Path to pretrained weights
        device: Device

    Returns:
        model: Model with loaded weights
    """
    state_dict = torch.load(pretrained_path, map_location=device)

    # Handle different state_dict formats
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # Only load CVE and DCC weights (CLIP weights are already loaded)
    cve_dcc_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("cve.") or key.startswith("dcc."):
            cve_dcc_state_dict[key] = value

    if cve_dcc_state_dict:
        missing, unexpected = model.load_state_dict(cve_dcc_state_dict, strict=False)
        print(f"Loaded CVE/DCC weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("No CVE/DCC weights found in checkpoint. Using default initialization.")

    return model


def interpolate_pos_embed(
        pos_embed: torch.Tensor,
        new_h: int,
        new_w: int,
) -> torch.Tensor:
    """
    Interpolate positional embeddings for different image sizes.

    Args:
        pos_embed: Original positional embedding [1 + H*W, D]
        new_h: New height in patches
        new_w: New width in patches

    Returns:
        new_pos_embed: Interpolated positional embedding
    """
    cls_token = pos_embed[:1]
    patch_embed = pos_embed[1:]

    old_num_patches = patch_embed.shape[0]
    old_h = old_w = int(math.sqrt(old_num_patches))

    dim = patch_embed.shape[1]

    patch_embed = patch_embed.reshape(1, old_h, old_w, dim).permute(0, 3, 1, 2)
    patch_embed = F.interpolate(
        patch_embed,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False
    )
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(-1, dim)

    new_pos_embed = torch.cat([cls_token, patch_embed], dim=0)

    return new_pos_embed


