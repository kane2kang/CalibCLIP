#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration utilities.
"""

import os
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with inheritance support.

    Args:
        config_path: Path to config file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Handle base config inheritance
    if "_BASE_" in config:
        base_path = config.pop("_BASE_")

        # Resolve relative path
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)

        # Load base config recursively
        base_config = load_config(base_path)

        # Merge configs (current overrides base)
        config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two config dictionaries.

    Args:
        base: Base configuration
        override: Override configuration (takes precedence)

    Returns:
        merged: Merged configuration
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:  # 只在目录非空时创建
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)



def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration dictionary.

    Returns:
        config: Default configuration
    """
    return {
        "model": {
            "name": "ViT-B/16",
            "image_size": [384, 128],
            "stride_size": 16,
            "embed_dim": 512,
            "pretrained_path": None,
            "vision": {
                "width": 768,
                "layers": 12,
                "patch_size": 16,
            },
            "text": {
                "context_length": 77,
                "vocab_size": 49408,
                "width": 512,
                "heads": 8,
                "layers": 12,
            },
        },
        "cve": {
            "enabled": True,
            "residual_coefficient": 0.1,
        },
        "dcc": {
            "enabled": True,
            "attention_threshold": None,
            "lambda_weight": 0.5,
            "top_k": 100,
        },
        "data": {
            "dataset": "cuhkpedes",
            "root": "./data/CUHK-PEDES",
            "json_path": None,
            "split": "test",
            "context_length": 77,
            "num_workers": 4,
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
        "eval": {
            "batch_size": 128,
            "num_workers": 4,
            "topk": [1, 5, 10],
            "compute_map": True,
            "compute_minp": True,
        },
        "output": {
            "dir": "./outputs",
            "save_features": False,
        },
        "seed": 42,
        "device": "cuda",
    }


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    required_fields = [
        ("model", "name"),
        ("model", "image_size"),
        ("data", "dataset"),
        ("data", "root"),
    ]

    for *path, field in required_fields:
        current = config
        for p in path:
            if p not in current:
                raise ValueError(f"Missing config section: {p}")
            current = current[p]
        if field not in current:
            raise ValueError(f"Missing required field: {'.'.join(path + [field])}")

    # Validate image size
    image_size = config["model"]["image_size"]
    if not (isinstance(image_size, (list, tuple)) and len(image_size) == 2):
        raise ValueError("model.image_size must be a list of [H, W]")

    # Validate lambda weight
    lambda_weight = config.get("dcc", {}).get("lambda_weight", 0.5)
    if not 0 <= lambda_weight <= 1:
        raise ValueError("dcc.lambda_weight must be in [0, 1]")

    # Validate residual coefficient
    rc = config.get("cve", {}).get("residual_coefficient", 0.1)
    if rc < 0:
        raise ValueError("cve.residual_coefficient must be non-negative")


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print configuration.

    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_config(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value}")
