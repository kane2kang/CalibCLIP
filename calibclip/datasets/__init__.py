#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for CalibCLIP.
"""

from typing import Optional, Tuple, Callable
from torch.utils.data import DataLoader

from .base_dataset import BaseRetrievalDataset
from .cuhkpedes import CUHKPEDESDataset
from .icfgpedes import ICFGPEDESDataset
from .rstp import RSTPDataset
from .tokenizer import SimpleTokenizer
from .transforms import build_transforms, TrainTransforms, TestTransforms

DATASET_REGISTRY = {
    "cuhkpedes": CUHKPEDESDataset,
    "cuhk-pedes": CUHKPEDESDataset,
    "icfgpedes": ICFGPEDESDataset,
    "icfg-pedes": ICFGPEDESDataset,
    "rstp": RSTPDataset,
}


def build_dataset(
        dataset_name: str,
        data_root: str,
        json_path: Optional[str] = None,
        split: str = "test",
        image_size: Tuple[int, int] = (384, 128),
        tokenizer: Optional[Callable] = None,
        max_length: int = 77,
        transform: Optional[Callable] = None,
) -> BaseRetrievalDataset:
    """
    Build dataset by name.

    Args:
        dataset_name: Name of dataset
        data_root: Root directory
        json_path: Path to annotation file
        split: Data split
        image_size: Target image size
        tokenizer: Text tokenizer
        max_length: Maximum token length
        transform: Image transform

    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_cls = DATASET_REGISTRY[dataset_name]

    return dataset_cls(
        data_root=data_root,
        json_path=json_path,
        split=split,
        image_size=image_size,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=transform,
    )


def build_dataloader(
        dataset_name: str,
        data_root: str,
        json_path: Optional[str] = None,
        split: str = "test",
        image_size: Tuple[int, int] = (384, 128),
        batch_size: int = 64,
        num_workers: int = 4,
        tokenizer: Optional[Callable] = None,
        max_length: int = 77,
        transform: Optional[Callable] = None,
        shuffle: bool = False,
) -> DataLoader:
    """
    Build dataloader.

    Args:
        dataset_name: Name of dataset
        data_root: Root directory
        json_path: Path to annotation file
        split: Data split
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers
        tokenizer: Text tokenizer
        max_length: Maximum token length
        transform: Image transform
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance
    """
    dataset = build_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        json_path=json_path,
        split=split,
        image_size=image_size,
        tokenizer=tokenizer,
        max_length=max_length,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )


__all__ = [
    "BaseRetrievalDataset",
    "CUHKPEDESDataset",
    "ICFGPEDESDataset",
    "RSTPDataset",
    "SimpleTokenizer",
    "build_dataset",
    "build_dataloader",
    "build_transforms",
    "TrainTransforms",
    "TestTransforms",
]
