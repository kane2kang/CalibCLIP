#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RSTPReid Dataset.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RSTPReid Dataset.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Callable

from .base_dataset import BaseRetrievalDataset


class RSTPDataset(BaseRetrievalDataset):
    """
    RSTPReid Dataset for text-based person retrieval.

    Dataset structure:
        RSTPReid/
        ├── imgs/
        │   └── ...
        └── data_captions.json

    Args:
        data_root: Dataset root directory
        split: Dataset split ("train", "val", "test")
        json_path: Path to annotation JSON file
        image_size: Image size (H, W)
        tokenizer: Text tokenizer
        max_length: Maximum text length
        transform: Image transforms
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
        self.json_path = json_path or os.path.join(data_root, "data_captions.json")
        super().__init__(
            data_root=data_root,
            split=split,
            image_size=image_size,
            tokenizer=tokenizer,
            max_length=max_length,
            transform=transform,
        )

    def _load_data(self) -> List[Dict]:
        """Load RSTPReid annotations."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        data = []
        for item in raw_data:
            if item.get("split", "test") != self.split:
                continue

            image_id = item["id"]
            image_path = item.get("img_path", item.get("file_path", ""))

            captions = item.get("captions", [])
            if isinstance(captions, str):
                captions = [captions]

            for cap_idx, caption in enumerate(captions):
                if caption.strip():
                    data.append({
                        "image_path": image_path,
                        "caption": caption,
                        "image_id": image_id,
                        "caption_id": f"{image_id}_{cap_idx}",
                    })

        print(f"[RSTPReid] Loaded {len(data)} samples for split '{self.split}'")
        return data
