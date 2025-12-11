#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image transforms for CalibCLIP.
"""

from typing import Tuple, Optional
from torchvision import transforms


class TrainTransforms:
    """Training transforms with augmentation."""

    def __init__(
            self,
            image_size: Tuple[int, int] = (384, 128),
            mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
            std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class TestTransforms:
    """Test/validation transforms without augmentation."""

    def __init__(
            self,
            image_size: Tuple[int, int] = (384, 128),
            mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
            std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711),
    ):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


def build_transforms(
        is_train: bool = False,
        image_size: Tuple[int, int] = (384, 128),
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    """
    Build image transforms.

    Args:
        is_train: Whether to use training transforms
        image_size: Target image size (H, W)
        mean: Normalization mean (default: CLIP ImageNet mean)
        std: Normalization std (default: CLIP ImageNet std)

    Returns:
        Transform function
    """
    # CLIP default normalization values
    if mean is None:
        mean = (0.48145466, 0.4578275, 0.40821073)
    if std is None:
        std = (0.26862954, 0.26130258, 0.27577711)

    if is_train:
        return TrainTransforms(image_size=image_size, mean=mean, std=std)
    else:
        return TestTransforms(image_size=image_size, mean=mean, std=std)


__all__ = [
    "TrainTransforms",
    "TestTransforms",
    "build_transforms",
]
