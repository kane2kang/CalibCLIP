#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for CalibCLIP.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from calibclip.models import build_calibclip
from calibclip.datasets import build_transforms
from calibclip.datasets.tokenizer import SimpleTokenizer
from calibclip.utils import visualize_attention, visualize_cve_analysis


def main():
    parser = argparse.ArgumentParser(description="CalibCLIP Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--text", type=str, required=True, help="Query text")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs", help="Output directory")
    parser.add_argument("--no_cve", action="store_true", help="Disable CVE")
    parser.add_argument("--no_dcc", action="store_true", help="Disable DCC")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    print("Building model...")
    image_size = (384, 128)
    model = build_calibclip(
        model_name="ViT-B/16",
        image_size=image_size,
        stride_size=16,
        device=device,
    )
    model.eval()

    # Build transforms and tokenizer
    transforms = build_transforms(image_size=image_size, is_train=False)
    tokenizer = SimpleTokenizer()

    # Load and process image
    print(f"Loading image: {args.image}")
    original_image = Image.open(args.image).convert("RGB")
    image_tensor = transforms(original_image).unsqueeze(0).to(device)

    # Tokenize text
    print(f"Query text: {args.text}")
    text_tokens = tokenizer(args.text, context_length=77).unsqueeze(0).to(device)

    # Forward pass
    apply_cve = not args.no_cve
    apply_dcc = not args.no_dcc

    print(f"CVE enabled: {apply_cve}, DCC enabled: {apply_dcc}")

    with torch.no_grad():
        # Encode image
        image_features, image_info = model.encode_image(image_tensor, apply_cve=apply_cve)

        # Encode text
        text_features, text_info = model.encode_text(text_tokens, apply_dcc=apply_dcc)

    # Compute similarity
    similarity = (image_features @ text_features.t()).item()
    print(f"\nSimilarity: {similarity:.4f}")

    # Get attention info
    cls_attention = image_info.get("cls_attention")

    if cls_attention is not None:
        cls_attention_np = cls_attention[0].cpu().numpy()

        # Get number of patches
        num_patches_h = model.h_patches
        num_patches_w = model.w_patches
        num_patches = (num_patches_h, num_patches_w)

        # Visualize attention
        print("\nVisualizing attention map...")
        original_np = np.array(original_image.resize((image_size[1], image_size[0])))

        fig = visualize_attention(
            image=original_np,
            attention=cls_attention_np,
            num_patches=num_patches,
            save_path=os.path.join(args.output_dir, "attention.png"),
            title=f"Query: {args.text[:50]}{'...' if len(args.text) > 50 else ''}",
        )
        print(f"Attention map saved to {args.output_dir}/attention.png")

        # Visualize CVE analysis if CVE was applied
        if apply_cve and "cve_info" in image_info:
            print("\nVisualizing CVE analysis...")
            cve_info = image_info["cve_info"]

            dominant_mask = cve_info.get("dominant_mask")
            context_mask = cve_info.get("context_mask")
            gate_weights = cve_info.get("gate_weights")

            if dominant_mask is not None and context_mask is not None:
                fig = visualize_cve_analysis(
                    image=original_np,
                    cls_attention=cls_attention_np,
                    dominant_mask=dominant_mask[0].cpu().numpy(),
                    context_mask=context_mask[0].cpu().numpy(),
                    gate_weights=gate_weights[0].cpu().numpy() if gate_weights is not None else None,
                    num_patches=num_patches,
                    save_path=os.path.join(args.output_dir, "cve_analysis.png"),
                )
                print(f"CVE analysis saved to {args.output_dir}/cve_analysis.png")

    # Print feature info
    print(f"\nImage feature shape: {image_features.shape}")
    print(f"Text feature shape: {text_features.shape}")
    print(f"Image feature norm: {image_features.norm(dim=-1).item():.4f}")
    print(f"Text feature norm: {text_features.norm(dim=-1).item():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
