#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for datasets.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch

from calibclip.datasets import SimpleTokenizer, tokenize, build_transforms


class TestTokenizer(unittest.TestCase):
    """Test SimpleTokenizer."""

    def setUp(self):
        self.tokenizer = SimpleTokenizer()

    def test_encode_decode(self):
        """Test encoding and decoding."""
        text = "a person wearing a red shirt"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)

        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        self.assertIn("person", decoded.lower())

    def test_call(self):
        """Test __call__ method."""
        texts = ["hello world", "a person walking"]
        result = self.tokenizer(texts, context_length=77)

        self.assertEqual(result.shape, (2, 77))
        self.assertEqual(result.dtype, torch.long)

    def test_single_text(self):
        """Test single text input."""
        text = "hello world"
        result = self.tokenizer(text, context_length=77)

        self.assertEqual(result.shape, (1, 77))

    def test_truncation(self):
        """Test long text truncation."""
        long_text = " ".join(["word"] * 100)
        result = self.tokenizer(long_text, context_length=77)

        self.assertEqual(result.shape, (1, 77))


class TestTokenizeFunction(unittest.TestCase):
    """Test tokenize function."""

    def test_tokenize(self):
        """Test tokenize function."""
        texts = ["hello world"]
        result = tokenize(texts, context_length=77)

        self.assertEqual(result.shape, (1, 77))
        self.assertEqual(result.dtype, torch.long)


class TestTransforms(unittest.TestCase):
    """Test image transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        transforms = build_transforms(
            image_size=(384, 128),
            is_train=True,
        )

        self.assertIsNotNone(transforms)

    def test_eval_transforms(self):
        """Test evaluation transforms."""
        transforms = build_transforms(
            image_size=(384, 128),
            is_train=False,
        )

        self.assertIsNotNone(transforms)


if __name__ == "__main__":
    unittest.main()
