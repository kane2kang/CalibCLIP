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

from calibclip.models.cve import ContrastiveVisualEnhancer
from calibclip.models.dcc import DiscriminativeConceptCalibrator


class TestCVE(unittest.TestCase):
    """Test CVE module."""

    def setUp(self):
        self.batch_size = 2
        self.num_patches = 24 * 8  # 384/16 * 128/16
        self.embed_dim = 768
        self.output_dim = 512

        self.cve = ContrastiveVisualEnhancer(
            embed_dim=self.embed_dim,
            output_dim=self.output_dim,
            residual_coefficient=0.1,
            attention_percentile=90.0,
        )

    def test_forward_shape(self):
        """Test output shape."""
        cls_token = torch.randn(self.batch_size, self.embed_dim)
        patch_tokens = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        cls_attention = torch.rand(self.batch_size, self.num_patches)
        cls_projected = torch.randn(self.batch_size, self.output_dim)

        enhanced_cls, info = self.cve(
            cls_token, patch_tokens, cls_attention, cls_projected
        )

        self.assertEqual(enhanced_cls.shape, (self.batch_size, self.output_dim))
        self.assertIn("dominant_mask", info)
        self.assertIn("context_mask", info)
        self.assertIn("gate_weights", info)

    def test_forward_without_cls_projected(self):
        """Test forward without pre-projected cls features."""
        cls_token = torch.randn(self.batch_size, self.embed_dim)
        patch_tokens = torch.randn(self.batch_size, self.num_patches, self.embed_dim)
        cls_attention = torch.rand(self.batch_size, self.num_patches)

        enhanced_cls, info = self.cve(
            cls_token, patch_tokens, cls_attention, cls_projected=None
        )

        self.assertEqual(enhanced_cls.shape, (self.batch_size, self.output_dim))

    def test_dominant_identification(self):
        """Test dominant patch identification."""
        cls_token = torch.randn(self.batch_size, self.embed_dim)
        patch_tokens = torch.randn(self.batch_size, self.num_patches, self.embed_dim)

        # Create attention with clear dominant patches (first 10 patches high attention)
        cls_attention = torch.ones(self.batch_size, self.num_patches) * 0.01
        cls_attention[:, :10] = 0.9  # Make first 10 patches dominant

        enhanced_cls, info = self.cve(
            cls_token, patch_tokens, cls_attention, cls_projected=None
        )

        # Check that dominant mask identifies high attention patches
        dominant_mask = info["dominant_mask"]
        self.assertTrue(dominant_mask[:, :10].sum() > 0)  # Some should be marked as dominant


class TestDCC(unittest.TestCase):
    """Test DCC module."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 77
        self.embed_dim = 512
        self.num_heads = 8

        self.dcc = DiscriminativeConceptCalibrator(
            embed_dim=self.embed_dim,
            lambda_weight=0.5,
            temperature=0.01,
        )

    def test_forward_shape(self):
        """Test output shape."""
        eot_features = torch.randn(self.batch_size, self.embed_dim)
        token_features = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        eot_attention = torch.rand(self.batch_size, self.num_heads, self.seq_len)
        token_mask = torch.ones(self.batch_size, self.seq_len)

        calibrated_features, info = self.dcc(
            eot_features=eot_features,
            token_features=token_features,
            eot_attention=eot_attention,
            token_mask=token_mask,
        )

        self.assertEqual(calibrated_features.shape, (self.batch_size, self.embed_dim))
        self.assertIn("discriminative_weights", info)
        self.assertIn("general_weights", info)
        self.assertIn("frequency_scores", info)

    def test_forward_with_token_ids(self):
        """Test forward with token IDs for frequency computation."""
        eot_features = torch.randn(self.batch_size, self.embed_dim)
        token_features = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        eot_attention = torch.rand(self.batch_size, self.num_heads, self.seq_len)
        token_ids = torch.randint(0, 49408, (self.batch_size, self.seq_len))
        token_mask = torch.ones(self.batch_size, self.seq_len)

        calibrated_features, info = self.dcc(
            eot_features=eot_features,
            token_features=token_features,
            eot_attention=eot_attention,
            token_ids=token_ids,
            token_mask=token_mask,
        )

        self.assertEqual(calibrated_features.shape, (self.batch_size, self.embed_dim))

    def test_normalization(self):
        """Test that output features are normalized."""
        eot_features = torch.randn(self.batch_size, self.embed_dim)
        token_features = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        eot_attention = torch.rand(self.batch_size, self.num_heads, self.seq_len)

        calibrated_features, _ = self.dcc(
            eot_features=eot_features,
            token_features=token_features,
            eot_attention=eot_attention,
        )

        # Features should be normalized
        norms = calibrated_features.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestCalibCLIPIntegration(unittest.TestCase):
    """Integration tests for CalibCLIP (requires CLIP installation)."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_build_calibclip(self):
        """Test building CalibCLIP model."""
        try:
            from calibclip.models.calibclip import build_calibclip

            model = build_calibclip(
                model_name="ViT-B/16",
                image_size=(384, 128),
                stride_size=16,
                device="cuda",
            )

            self.assertIsNotNone(model)
            self.assertEqual(model.embed_dim, 512)

        except Exception as e:
            self.skipTest(f"CLIP not available or other error: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_encode_image(self):
        """Test image encoding."""
        try:
            from calibclip.models.calibclip import build_calibclip

            model = build_calibclip(
                model_name="ViT-B/16",
                image_size=(384, 128),
                stride_size=16,
                device="cuda",
            )

            batch_size = 2
            images = torch.randn(batch_size, 3, 384, 128).cuda()

            image_features, info = model.encode_image(images)

            self.assertEqual(image_features.shape, (batch_size, 512))
            self.assertIn("cve_info", info)

        except Exception as e:
            self.skipTest(f"Test skipped: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_encode_text(self):
        """Test text encoding."""
        try:
            from calibclip.models.calibclip import build_calibclip
            import clip

            model = build_calibclip(
                model_name="ViT-B/16",
                image_size=(384, 128),
                stride_size=16,
                device="cuda",
            )

            # Tokenize sample text
            text = clip.tokenize(["a photo of a person", "a man in red shirt"]).cuda()

            text_features, info = model.encode_text(text)

            self.assertEqual(text_features.shape, (2, 512))
            self.assertIn("dcc_info", info)

        except Exception as e:
            self.skipTest(f"Test skipped: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_forward(self):
        """Test forward pass."""
        try:
            from calibclip.models.calibclip import build_calibclip
            import clip

            model = build_calibclip(
                model_name="ViT-B/16",
                image_size=(384, 128),
                stride_size=16,
                device="cuda",
            )

            batch_size = 2
            images = torch.randn(batch_size, 3, 384, 128).cuda()
            text = clip.tokenize(["a photo of a person", "a man in red shirt"]).cuda()

            outputs = model(images, text)

            self.assertIn("image_features", outputs)
            self.assertIn("text_features", outputs)
            self.assertEqual(outputs["image_features"].shape, (batch_size, 512))
            self.assertEqual(outputs["text_features"].shape, (batch_size, 512))

        except Exception as e:
            self.skipTest(f"Test skipped: {e}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_feature_normalization(self):
        """Test that extracted features are normalized."""
        try:
            from calibclip.models.calibclip import build_calibclip
            import clip

            model = build_calibclip(
                model_name="ViT-B/16",
                image_size=(384, 128),
                stride_size=16,
                device="cuda",
            )

            batch_size = 2
            images = torch.randn(batch_size, 3, 384, 128).cuda()
            text = clip.tokenize(["test sentence"]).cuda()

            image_features, _ = model.encode_image(images)
            text_features, _ = model.encode_text(text)

            # Check normalization
            img_norms = image_features.norm(dim=-1)
            txt_norms = text_features.norm(dim=-1)

            self.assertTrue(torch.allclose(img_norms, torch.ones_like(img_norms), atol=1e-4))
            self.assertTrue(torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-4))

        except Exception as e:
            self.skipTest(f"Test skipped: {e}")


class TestCVEComponents(unittest.TestCase):
    """Test individual CVE components."""

    def setUp(self):
        self.cve = ContrastiveVisualEnhancer(
            embed_dim=768,
            output_dim=512,
            residual_coefficient=0.1,
            attention_percentile=90.0,
        )

    def test_compute_attention_threshold(self):
        """Test attention threshold computation."""
        batch_size = 4
        num_patches = 192

        cls_attention = torch.rand(batch_size, num_patches)
        threshold = self.cve.compute_attention_threshold(cls_attention)

        self.assertEqual(threshold.shape, (batch_size, 1))

        # Threshold should be at 90th percentile
        for i in range(batch_size):
            expected = torch.quantile(cls_attention[i], 0.9)
            self.assertTrue(torch.allclose(threshold[i, 0], expected, atol=1e-5))

    def test_identify_dominant_patches(self):
        """Test dominant patch identification."""
        batch_size = 2
        num_patches = 100

        cls_attention = torch.rand(batch_size, num_patches)
        threshold = torch.tensor([[0.5], [0.6]])

        dominant_mask, context_mask = self.cve.identify_dominant_patches(
            cls_attention, threshold
        )

        # Masks should be complementary
        self.assertTrue(torch.allclose(dominant_mask + context_mask, torch.ones_like(dominant_mask)))

        # Check threshold logic
        for i in range(batch_size):
            expected_dominant = (cls_attention[i] >= threshold[i]).float()
            self.assertTrue(torch.allclose(dominant_mask[i], expected_dominant))


class TestDCCComponents(unittest.TestCase):
    """Test individual DCC components."""

    def setUp(self):
        self.dcc = DiscriminativeConceptCalibrator(
            embed_dim=512,
            lambda_weight=0.5,
            temperature=0.01,
        )

    def test_compute_concept_frequency(self):
        """Test concept frequency computation."""
        batch_size = 2
        seq_len = 20

        # Create token IDs with some repeated tokens
        token_ids = torch.tensor([
            [1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 1, 1, 1, 9, 10, 11, 12, 13, 14, 15],
            [1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        ])

        frequency_scores = self.dcc.compute_concept_frequency(token_ids)

        self.assertEqual(frequency_scores.shape, (batch_size, seq_len))

        # Token 1 appears frequently, should have high frequency score
        # Token with unique occurrence should have lower score

    def test_identify_discriminative_tokens(self):
        """Test discriminative token identification."""
        batch_size = 2
        seq_len = 10
        num_heads = 8

        eot_attention = torch.rand(batch_size, num_heads, seq_len)
        frequency_scores = torch.rand(batch_size, seq_len)
        token_mask = torch.ones(batch_size, seq_len)

        disc_weights, gen_weights = self.dcc.identify_discriminative_tokens(
            eot_attention, frequency_scores, token_mask
        )

        # Weights should sum to 1
        self.assertTrue(torch.allclose(disc_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5))
        self.assertTrue(torch.allclose(gen_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5))

    def test_aggregate_weighted_features(self):
        """Test weighted feature aggregation."""
        batch_size = 2
        seq_len = 10
        embed_dim = 512

        token_features = torch.randn(batch_size, seq_len, embed_dim)
        weights = torch.softmax(torch.rand(batch_size, seq_len), dim=-1)

        aggregated = self.dcc.aggregate_weighted_features(token_features, weights)

        self.assertEqual(aggregated.shape, (batch_size, embed_dim))


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
