#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for CalibCLIP models.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch

from calibclip.models import CalibCLIP, build_calibclip
from calibclip.models.cve import ContextualVisualEnhancement
from calibclip.models.dcc import DiscriminativeContextCalibration


class TestCVE(unittest.TestCase):
    """Test CVE module."""

    def setUp(self):
        self.batch_size = 2
        self.num_patches = 24 * 8  # 384/16 * 128/16
        self.embed_dim = 768
        self.output_dim = 512

        self.cve = ContextualVisualEnhancement(
            embed_dim=self.embed_dim,
            output_dim=self.output_dim,
            residual_coefficient=0.1,
        )

    def test_forward_shape(self):
        """Test output shape."""
        cls_token = torch.randn(self.batch_size, self.embed_dim)
        patch_tokens = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        cls_attention = torch.rand(self.batch_size, self.num_patches)
        text_features = torch.randn(self.batch_size, self.output_dim)

        enhanced_cls, info = self.cve(
            cls_token, patch_tokens, cls_attention, text_features
        )

        self.assertEqual(enhanced_cls.shape, (self.batch_size, self.output_dim))
        self.assertIn("foreground_mask", info)
        self.assertIn("dominant_mask", info)
        self.assertIn("context_weights", info)

    def test_forward_without_text(self):
        """Test forward without text features."""
        cls_token = torch.randn(self.batch_size, self.embed_dim)
        patch_tokens = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        cls_attention = torch.rand(self.batch_size, self.num_patches)

        enhanced_cls, info = self.cve(
            cls_token, patch_tokens, cls_attention, text_features=None
        )

        self.assertEqual(enhanced_cls.shape, (self.batch_size, self.output_dim))


class TestDCC(unittest.TestCase):
    """Test DCC module."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 77
        self.embed_dim = 512
        self.num_heads = 8

        self.dcc = DiscriminativeContextCalibration(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attention_threshold=None,
            top_k=100,
        )

    def test_forward_shape(self):
        """Test output shape."""
        eot_features = torch.randn(self.batch_size, self.embed_dim)
        token_features = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        attention_weights = torch.rand(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        attention_mask = torch.ones(self.batch_size, self.seq_len)

        disc_features, info = self.dcc(
            eot_features, token_features, attention_weights, attention_mask
        )

        self.assertEqual(disc_features.shape, (self.batch_size, self.embed_dim))
        self.assertIn("discriminative_mask", info)
        self.assertIn("selected_indices", info)

    def test_discriminative_selection(self):
        """Test discriminative token selection."""
        # Create attention weights with clear pattern
        attention_weights = torch.zeros(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        # Make first few tokens highly attended
        attention_weights[:, :, :, :5] = 1.0
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        eot_features = torch.randn(self.batch_size, self.embed_dim)
        token_features = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        attention_mask = torch.ones(self.batch_size, self.seq_len)

        disc_features, info = self.dcc(
            eot_features, token_features, attention_weights, attention_mask
        )

        # Discriminative mask should exclude highly attended tokens
        mask = info["discriminative_mask"]
        self.assertTrue(mask[:, :5].sum() < mask[:, 5:].sum())


class TestCalibCLIP(unittest.TestCase):
    """Test CalibCLIP model."""

    def setUp(self):
        self.batch_size = 2
        self.image_size = (384, 128)
        self.context_length = 77

        self.model = CalibCLIP(
            image_resolution=self.image_size,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            vision_stride_size=16,
            context_length=self.context_length,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
            embed_dim=512,
        )

    def test_encode_image(self):
        """Test image encoding."""
        images = torch.randn(self.batch_size, 3, *self.image_size)

        cls_features, patch_features, info = self.model.encode_image(images)

        self.assertEqual(cls_features.shape, (self.batch_size, 512))
        self.assertIsNotNone(patch_features)

    def test_encode_text(self):
        """Test text encoding."""
        text = torch.randint(0, 49408, (self.batch_size, self.context_length))

        eot_features, disc_features, info = self.model.encode_text(text)

        self.assertEqual(eot_features.shape, (self.batch_size, 512))

    def test_forward(self):
        """Test forward pass."""
        images = torch.randn(self.batch_size, 3, *self.image_size)
        text = torch.randint(0, 49408, (self.batch_size, self.context_length))

        similarity = self.model(images, text)

        self.assertEqual(similarity.shape, (self.batch_size, self.batch_size))

    def test_extract_features(self):
        """Test feature extraction."""
        images = torch.randn(self.batch_size, 3, *self.image_size)
        text = torch.randint(0, 49408, (self.batch_size, self.context_length))

        image_features = self.model.extract_image_features(images)
        text_features, disc_features = self.model.extract_text_features(text)

        self.assertEqual(image_features.shape, (self.batch_size, 512))
        self.assertEqual(text_features.shape, (self.batch_size, 512))

        # Features should be normalized
        norms = image_features.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestBuildCalibCLIP(unittest.TestCase):
    """Test model building function."""

    def test_build_vit_b16(self):
        """Test building ViT-B/16."""
        model = build_calibclip(
            name="ViT-B/16",
            image_size=(384, 128),
            stride_size=16,
        )

        self.assertIsInstance(model, CalibCLIP)
        self.assertEqual(model.embed_dim, 512)

    def test_build_vit_b32(self):
        """Test building ViT-B/32."""
        model = build_calibclip(
            name="ViT-B/32",
            image_size=(384, 128),
            stride_size=32,
        )

        self.assertIsInstance(model, CalibCLIP)
        self.assertEqual(model.embed_dim, 512)

    def test_build_invalid_name(self):
        """Test building with invalid name."""
        with self.assertRaises(ValueError):
            build_calibclip(name="InvalidModel")


if __name__ == "__main__":
    unittest.main()
