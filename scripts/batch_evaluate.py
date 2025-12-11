#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch evaluation script for ablation studies.
"""

import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from calibclip.models import build_calibclip
from calibclip.datasets import (
    CUHKPEDESDataset,
    ICFGPEDESDataset,
    RSTPDataset,
    build_transforms,
)
from calibclip.datasets.tokenizer import SimpleTokenizer
from calibclip.evaluation import Evaluator
from calibclip.utils import load_config, setup_logger


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


def extract_features(model, dataset, device, batch_size, num_workers, apply_cve, apply_dcc):
    """Extract image and text features from dataset."""
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_image_features = []
    all_text_features = []
    all_pids = []
    all_image_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            pids = batch["pid"]
            image_ids = batch.get("image_id", batch.get("img_path", range(len(pids))))

            # Encode
            outputs = model(
                images=images,
                text=text_tokens,
                apply_cve=apply_cve,
                apply_dcc=apply_dcc,
            )

            all_image_features.append(outputs["image_features"].cpu())
            all_text_features.append(outputs["text_features"].cpu())
            all_pids.extend(pids.tolist() if torch.is_tensor(pids) else pids)
            all_image_ids.extend(image_ids)

    image_features = torch.cat(all_image_features, dim=0)
    text_features = torch.cat(all_text_features, dim=0)

    return image_features, text_features, all_pids, all_image_ids


def run_single_evaluation(
        model,
        dataset,
        device,
        batch_size,
        num_workers,
        apply_cve,
        apply_dcc,
        topk,
        logger,
):
    """Run a single evaluation configuration."""
    # Extract features
    image_features, text_features, pids, image_ids = extract_features(
        model, dataset, device, batch_size, num_workers, apply_cve, apply_dcc
    )

    # Get unique images
    unique_image_ids = []
    unique_indices = []
    seen_ids = set()
    for i, img_id in enumerate(image_ids):
        img_id_str = str(img_id)
        if img_id_str not in seen_ids:
            seen_ids.add(img_id_str)
            unique_image_ids.append(img_id)
            unique_indices.append(i)

    unique_image_features = image_features[unique_indices]
    unique_pids = [pids[i] for i in unique_indices]

    # Build ground truth for text-to-image
    text2image_gt = {}
    for i, pid in enumerate(pids):
        matching_indices = [j for j, p in enumerate(unique_pids) if p == pid]
        text2image_gt[i] = matching_indices

    # Compute similarity
    similarity = text_features @ unique_image_features.t()

    # Evaluate
    evaluator = Evaluator(topk=topk, compute_map=True, compute_minp=True)
    metrics = evaluator.evaluate(similarity, text2image_gt, query_type="text")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for CalibCLIP")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default="./ablation_outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num workers")
    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logger
    logger = setup_logger("calibclip", save_dir=output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Load config
    config = load_config(args.config)
    if args.batch_size:
        config["eval"]["batch_size"] = args.batch_size
    if args.num_workers:
        config["eval"]["num_workers"] = args.num_workers

    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # Build model
    logger.info("Building model...")
    model = build_calibclip(
        model_name=config["model"]["name"],
        image_size=tuple(config["model"]["image_size"]),
        stride_size=config["model"]["stride_size"],
        device=args.device,
        cve_config=config.get("cve", {}),
        dcc_config=config.get("dcc", {}),
    )

    # Build transforms and tokenizer
    transforms = build_transforms(
        image_size=tuple(config["model"]["image_size"]),
        is_train=False,
        mean=config["data"].get("mean"),
        std=config["data"].get("std"),
    )
    tokenizer = SimpleTokenizer()

    # Build dataset
    logger.info("Loading dataset...")
    dataset = get_dataset(config, transforms, tokenizer)

    # Evaluation settings
    batch_size = config["eval"]["batch_size"]
    num_workers = config["eval"]["num_workers"]
    topk = config["eval"]["topk"]

    # Store all results
    all_results = {}

    # ==========================================
    # 1. Component Ablation Study
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("1. Component Ablation Study")
    logger.info("=" * 60)

    ablation_configs = [
        {"cve": False, "dcc": False, "name": "Baseline (CLIP)"},
        {"cve": True, "dcc": False, "name": "CVE Only"},
        {"cve": False, "dcc": True, "name": "DCC Only"},
        {"cve": True, "dcc": True, "name": "CalibCLIP (CVE + DCC)"},
    ]

    component_results = []
    for ablation in ablation_configs:
        logger.info(f"\nRunning: {ablation['name']}")
        logger.info(f"  CVE: {ablation['cve']}, DCC: {ablation['dcc']}")

        metrics = run_single_evaluation(
            model=model,
            dataset=dataset,
            device=args.device,
            batch_size=batch_size,
            num_workers=num_workers,
            apply_cve=ablation["cve"],
            apply_dcc=ablation["dcc"],
            topk=topk,
            logger=logger,
        )

        result_entry = {
            "name": ablation["name"],
            "cve": ablation["cve"],
            "dcc": ablation["dcc"],
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        component_results.append(result_entry)

        logger.info(f"Results for {ablation['name']}:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.2f}")

    all_results["component_ablation"] = component_results

    # ==========================================
    # 2. Lambda Weight Ablation Study
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("2. Lambda Weight Ablation Study (with CVE + DCC)")
    logger.info("=" * 60)

    lambda_weights = [0.1, 0.3, 0.5, 0.7, 0.9]

    lambda_results = []
    for lw in lambda_weights:
        logger.info(f"\nRunning: Lambda = {lw}")

        # Rebuild model with different lambda weight
        model_temp = build_calibclip(
            model_name=config["model"]["name"],
            image_size=tuple(config["model"]["image_size"]),
            stride_size=config["model"]["stride_size"],
            device=args.device,
            cve_config=config.get("cve", {}),
            dcc_config={"lambda_weight": lw,
                        **{k: v for k, v in config.get("dcc", {}).items() if k != "lambda_weight"}},
        )

        metrics = run_single_evaluation(
            model=model_temp,
            dataset=dataset,
            device=args.device,
            batch_size=batch_size,
            num_workers=num_workers,
            apply_cve=True,
            apply_dcc=True,
            topk=topk,
            logger=logger,
        )

        result_entry = {
            "lambda_weight": lw,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        lambda_results.append(result_entry)

        logger.info(f"Results for Lambda = {lw}:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.2f}")

        del model_temp
        torch.cuda.empty_cache()

    all_results["lambda_ablation"] = lambda_results

    # ==========================================
    # 3. CVE Residual Coefficient Ablation
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("3. CVE Residual Coefficient Ablation")
    logger.info("=" * 60)

    residual_coefficients = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

    residual_results = []
    for rc in residual_coefficients:
        logger.info(f"\nRunning: Residual Coefficient = {rc}")

        model_temp = build_calibclip(
            model_name=config["model"]["name"],
            image_size=tuple(config["model"]["image_size"]),
            stride_size=config["model"]["stride_size"],
            device=args.device,
            cve_config={"residual_coefficient": rc},
            dcc_config=config.get("dcc", {}),
        )

        metrics = run_single_evaluation(
            model=model_temp,
            dataset=dataset,
            device=args.device,
            batch_size=batch_size,
            num_workers=num_workers,
            apply_cve=True,
            apply_dcc=True,
            topk=topk,
            logger=logger,
        )

        result_entry = {
            "residual_coefficient": rc,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        residual_results.append(result_entry)

        logger.info(f"Results for Residual Coefficient = {rc}:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.2f}")

        del model_temp
        torch.cuda.empty_cache()

    all_results["residual_ablation"] = residual_results

    # ==========================================
    # Save Results
    # ==========================================
    json_path = os.path.join(output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info("\nComponent Ablation:")
    for result in component_results:
        m = result["metrics"]
        r1 = m.get("R@1", m.get("recall@1", 0))
        mAP = m.get("mAP", 0)
        logger.info(f"  {result['name']:<25}: R@1={r1:.2f}, mAP={mAP:.2f}")

    logger.info("\nBest Lambda Weight:")
    best_lambda = max(lambda_results, key=lambda x: x["metrics"].get("R@1", x["metrics"].get("recall@1", 0)))
    logger.info(
        f"  Lambda={best_lambda['lambda_weight']}: R@1={best_lambda['metrics'].get('R@1', best_lambda['metrics'].get('recall@1', 0)):.2f}")

    logger.info("\nBest Residual Coefficient:")
    best_residual = max(residual_results, key=lambda x: x["metrics"].get("R@1", x["metrics"].get("recall@1", 0)))
    logger.info(
        f"  RC={best_residual['residual_coefficient']}: R@1={best_residual['metrics'].get('R@1', best_residual['metrics'].get('recall@1', 0)):.2f}")

    logger.info("\nAblation study completed!")


if __name__ == "__main__":
    main()
