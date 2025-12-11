#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base dataset class for text-image retrieval.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class BaseRetrievalDataset(Dataset, ABC):
    """
    Base class for text-image retrieval datasets.
    """

    def __init__(
            self,
            data_root: str,
            split: str = "test",
            image_size: Tuple[int, int] = (384, 128),
            tokenizer: Optional[Callable] = None,
            max_length: int = 77,
            transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_root: Root directory of dataset
            split: Data split ('train', 'val', 'test')
            image_size: Target image size (H, W)
            tokenizer: Text tokenizer
            max_length: Maximum token length
            transform: Image transform (if None, default transform is used)
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Setup transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform()

        # Load data
        self.data = self._load_data()

    def _default_transform(self) -> Callable:
        """Create default image transform."""
        return T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    @abstractmethod
    def _load_data(self) -> List[Dict]:
        """
        Load dataset annotations.

        Returns:
            List of data items, each containing:
                - image_path: Path to image
                - caption: Text caption
                - image_id: Unique image identifier
                - caption_id: Unique caption identifier (optional)
        """
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Load image
        image_path = os.path.join(self.data_root, item["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Tokenize caption
        caption = item["caption"]
        if self.tokenizer is not None:
            text_tokens = self.tokenizer(caption, max_length=self.max_length)
            if isinstance(text_tokens, list):
                text_tokens = torch.tensor(text_tokens, dtype=torch.long)
        else:
            text_tokens = caption

        return {
            "images": image,
            "text_tokens": text_tokens,
            "captions": caption,
            "image_ids": item["image_id"],
            "caption_ids": item.get("caption_id", item["image_id"]),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function."""
        images = torch.stack([item["images"] for item in batch])

        # Handle text tokens
        text_tokens = [item["text_tokens"] for item in batch]
        if isinstance(text_tokens[0], torch.Tensor):
            text_tokens = torch.stack(text_tokens)

        return {
            "images": images,
            "text_tokens": text_tokens,
            "captions": [item["captions"] for item in batch],
            "image_ids": [item["image_ids"] for item in batch],
            "caption_ids": [item["caption_ids"] for item in batch],
        }
