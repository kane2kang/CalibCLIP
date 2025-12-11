#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CUHK-PEDES dataset for person re-identification.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Callable

from .base_dataset import BaseRetrievalDataset


class CUHKPEDESDataset(BaseRetrievalDataset):
    """
    CUHK-PEDES dataset.

    Directory structure:
        data_root/
            imgs/
                CUHK01/
                CUHK03/
                Market/
                ...
            reid_raw.json
    """

    def __init__(
            self,
            data_root: str,
            json_path: Optional[str] = None,
            split: str = "test",
            image_size: Tuple[int, int] = (384, 128),
            tokenizer: Optional[Callable] = None,
            max_length: int = 77,
            transform: Optional[Callable] = None,
    ):
        self.json_path = json_path or os.path.join(data_root, "reid_raw.json")
        super().__init__(
            data_root=data_root,
            split=split,
            image_size=image_size,
            tokenizer=tokenizer,
            max_length=max_length,
            transform=transform,
        )

    def _load_data(self) -> List[Dict]:
        """Load CUHK-PEDES annotations."""
        with open(self.json_path, "r") as f:
            raw_data = json.load(f)

        data = []
        for item in raw_data:
            if item["split"] != self.split:
                continue

            image_path = item["file_path"]
            image_id = item["id"]

            # Each image can have multiple captions
            for cap_idx, caption in enumerate(item["captions"]):
                data.append({
                    "image_path": image_path,
                    "caption": caption,
                    "image_id": image_id,
                    "caption_id": f"{image_id}_{cap_idx}",
                })

        return data
