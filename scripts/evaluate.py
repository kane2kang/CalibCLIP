#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation script for CalibCLIP.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibclip.models import build_calibclip
from calibclip.datasets import (
    CUHKPEDESDataset,
    ICFGPEDESDataset,
    RSTPDataset,
    build_transforms,
)
from calibclip.datasets.tokenizer import SimpleTokenizer
from calibclip.evaluation import Evaluator
from calibclip.utils.config import load_config, save_config, print_config
from calibclip.utils.logger import setup_logger


def get_dataset(config, transforms, tokenizer):
    """Get dataset based on config."""
    dataset_name = config["data"]["dataset"].lower()

    dataset_cls = {
        "cuhkpedes": CUHKPEDESDataset,
        "icfgpedes": ICFGPEDESDataset,
        "rstp": RSTPDataset,
    }

    if dataset_name not in dataset_cls:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset_cls[dataset_name](
        root=config["data"]["root"],
        split=config["data"].get("split", "test"),
        json_path=config["data"].get("json_path"),
        transforms=transforms,
        tokenizer=tokenizer,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CalibCLIP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cuhkpedes.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--no-cve",
        action="store_true",
        help="Disable CVE module",
    )
    parser.add_argument(
        "--no-dcc",
        action="store_true",
        help="Disable DCC module",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    if args.batch_size:
        config["eval"]["batch_size"] = args.batch_size

    # Setup output directory
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger("calibclip", save_dir=str(output_dir), filename="evaluate.log")
    logger.info(f"Config loaded from: {args.config}")
    print_config(config)

    # Save config
    save_config(config, str(output_dir / "config.yaml"))

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building model...")
    model = build_calibclip(
        model_name=config["model"]["name"],
        image_size=tuple(config["model"]["image_size"]),
        stride_size=config["model"]["stride_size"],
        device=device,
        cve_config=config.get("cve", {}),
        dcc_config=config.get("dcc", {}),
    )
    model.eval()

    # Build tokenizer and transforms
    tokenizer = SimpleTokenizer()
    transforms = build_transforms(
        image_size=tuple(config["model"]["image_size"]),
        is_train=False,
        mean=config["data"].get("mean"),
        std=config["data"].get("std"),
    )

    # Build dataset and dataloader
    logger.info("Building dataset...")
    dataset = get_dataset(config, transforms, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["eval"].get("num_workers", 4),
        pin_memory=True,
    )

    # Build evaluator
    evaluator = Evaluator(
        topk=config["eval"].get("topk", [1, 5, 10]),
        compute_map=config["eval"].get("compute_map", True),
        compute_minp=config["eval"].get("compute_minp", True),
    )

    # Extract features
    logger.info("Extracting features...")
    apply_cve = not args.no_cve
    apply_dcc = not args.no_dcc

    logger.info(f"CVE enabled: {apply_cve}, DCC enabled: {apply_dcc}")

    all_image_features = []
    all_text_features = []
    all_pids = []
    all_image_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            pids = batch["pid"]
            image_ids = batch.get("image_id", batch.get("img_path", list(range(len(pids)))))

            # Encode images
            image_features, _ = model.encode_image(images, apply_cve=apply_cve)

            # Encode text
            text_features, _ = model.encode_text(text_tokens, apply_dcc=apply_dcc)

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
            all_pids.extend(pids.tolist() if torch.is_tensor(pids) else pids)
            all_image_ids.extend(image_ids if isinstance(image_ids, list) else image_ids.tolist())

    # Concatenate features
    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    logger.info(f"Image features shape: {image_features.shape}")
    logger.info(f"Text features shape: {text_features.shape}")

    # Remove duplicate images (keep unique)
    unique_image_ids = []
    unique_indices = []
    seen_ids = set()
    for i, img_id in enumerate(all_image_ids):
        img_id_str = str(img_id)
        if img_id_str not in seen_ids:
            seen_ids.add(img_id_str)
            unique_image_ids.append(img_id)
            unique_indices.append(i)

    unique_image_features = image_features[unique_indices]
    unique_pids = [all_pids[i] for i in unique_indices]

    logger.info(f"Unique images: {len(unique_image_ids)}")
    logger.info(f"Total captions: {len(all_pids)}")

    # Build ground truth mapping for text-to-image
    text2image_gt = {}
    for i, pid in enumerate(all_pids):
        matching_indices = [j for j, p in enumerate(unique_pids) if p == pid]
        text2image_gt[i] = matching_indices

    # Compute similarity matrix
    logger.info("Computing similarity matrix...")
    similarity = text_features @ unique_image_features.t()

    # Evaluate text-to-image retrieval
    logger.info("Evaluating text-to-image retrieval...")
    t2i_metrics = evaluator.evaluate(
        similarity_matrix=similarity,
        ground_truth=text2image_gt,
        query_type="text",
    )

    # Print results
    logger.info("=" * 50)
    logger.info("Text-to-Image Retrieval Results:")
    logger.info("=" * 50)
    for k, v in t2i_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Image-to-text retrieval
    image2text_gt = {}
    for i, pid in enumerate(unique_pids):
        matching_captions = [j for j, p in enumerate(all_pids) if p == pid]
        image2text_gt[i] = matching_captions

    i2t_similarity = unique_image_features @ text_features.t()

    logger.info("Evaluating image-to-text retrieval...")
    i2t_metrics = evaluator.evaluate(
        similarity_matrix=i2t_similarity,
        ground_truth=image2text_gt,
        query_type="image",
    )

    logger.info("=" * 50)
    logger.info("Image-to-Text Retrieval Results:")
    logger.info("=" * 50)
    for k, v in i2t_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Save results
    results = {
        "text_to_image": {k: float(v) if isinstance(v, (int, float)) else v for k, v in t2i_metrics.items()},
        "image_to_text": {k: float(v) if isinstance(v, (int, float)) else v for k, v in i2t_metrics.items()},
        "config": config,
        "cve_enabled": apply_cve,
        "dcc_enabled": apply_dcc,
        "num_images": len(unique_image_ids),
        "num_captions": len(all_pids),
    }

    results_path = output_dir / "results.pth"
    torch.save(results, results_path)
    logger.info(f"Results saved to {results_path}")

    # Also save as text
    results_txt = output_dir / "results.txt"
    with open(results_txt, "w") as f:
        f.write("CalibCLIP Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {config['data']['dataset']}\n")
        f.write(f"CVE enabled: {apply_cve}\n")
        f.write(f"DCC enabled: {apply_dcc}\n")
        f.write(f"Unique images: {len(unique_image_ids)}\n")
        f.write(f"Total captions: {len(all_pids)}\n\n")

        f.write("Text-to-Image Retrieval:\n")
        f.write("-" * 30 + "\n")
        for k, v in t2i_metrics.items():
            f.write(f"  {k}: {v:.4f if isinstance(v, float) else v}\n")

        f.write("\nImage-to-Text Retrieval:\n")
        f.write("-" * 30 + "\n")
        for k, v in i2t_metrics.items():
            f.write(f"  {k}: {v:.4f if isinstance(v, float) else v}\n")

    logger.info(f"Text results saved to {results_txt}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()

