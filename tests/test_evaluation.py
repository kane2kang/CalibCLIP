#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for evaluation metrics.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch

from calibclip.evaluation import compute_metrics, compute_rank, compute_ap
from calibclip.evaluation import RetrievalEngine


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def test_compute_rank(self):
        """Test Rank-K computation."""
        # Perfect retrieval case
        similarity = np.array([
            [1.0, 0.5, 0.3],  # Query 0 matches Gallery 0
            [0.3, 1.0, 0.5],  # Query 1 matches Gallery 1
        ])
        query_pids = np.array([0, 1])
        gallery_pids = np.array([0, 1, 2])

        results = compute_rank(similarity, query_pids, gallery_pids, topk=[1, 2])

        self.assertEqual(results["R@1"], 100.0)
        self.assertEqual(results["R@2"], 100.0)

    def test_compute_rank_partial(self):
        """Test Rank-K with partial matches."""
        # Query 0 has wrong top-1 but correct top-2
        similarity = np.array([
            [0.5, 1.0, 0.3],  # Query 0 (pid=0) ranks Gallery 1 first
            [0.3, 1.0, 0.5],  # Query 1 (pid=1) matches Gallery 1
        ])
        query_pids = np.array([0, 1])
        gallery_pids = np.array([0, 1, 2])

        results = compute_rank(similarity, query_pids, gallery_pids, topk=[1, 2])

        self.assertEqual(results["R@1"], 50.0)  # Only 1/2 correct at R@1
        self.assertEqual(results["R@2"], 100.0)  # Both correct at R@2

    def test_compute_ap(self):
        """Test Average Precision computation."""
        # Perfect ranking
        similarity = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        query_pid = 0
        gallery_pids = np.array([0, 0, 1, 1, 1])

        ap = compute_ap(similarity, query_pid, gallery_pids)

        # AP should be 1.0 for perfect ranking of positives
        self.assertEqual(ap, 1.0)

    def test_compute_ap_imperfect(self):
        """Test AP with imperfect ranking."""
        similarity = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        query_pid = 0
        gallery_pids = np.array([1, 0, 0, 1, 1])  # Positives at index 1, 2

        ap = compute_ap(similarity, query_pid, gallery_pids)

        # AP = (1/2 + 2/3) / 2 = 0.583...
        self.assertAlmostEqual(ap, (0.5 + 2 / 3) / 2, places=5)

    def test_compute_metrics(self):
        """Test full metrics computation."""
        np.random.seed(42)
        similarity = np.random.rand(10, 20)
        query_pids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        gallery_pids = np.array([0, 1, 2, 3, 4] * 4)

        metrics = compute_metrics(
            similarity, query_pids, gallery_pids,
            topk=[1, 5, 10],
            compute_map_flag=True,
            compute_minp_flag=True,
        )

        self.assertIn("R@1", metrics)
        self.assertIn("R@5", metrics)
        self.assertIn("R@10", metrics)
        self.assertIn("mAP", metrics)
        self.assertIn("mINP", metrics)


class TestRetrievalEngine(unittest.TestCase):
    """Test RetrievalEngine."""

    def setUp(self):
        self.engine = RetrievalEngine(lambda_weight=0.5)

    def test_compute_similarity(self):
        """Test similarity computation."""
        query = torch.randn(5, 512)
        gallery = torch.randn(10, 512)

        similarity = self.engine.compute_similarity(query, gallery)

        self.assertEqual(similarity.shape, (5, 10))

    def test_two_stage_retrieval(self):
        """Test two-stage retrieval."""
        query = torch.randn(5, 512)
        query_disc = torch.randn(5, 512)
        gallery = torch.randn(10, 512)

        similarity = self.engine.two_stage_retrieval(query, query_disc, gallery)

        self.assertEqual(similarity.shape, (5, 10))

    def test_retrieve(self):
        """Test retrieve method."""
        query = torch.randn(5, 512)
        gallery = torch.randn(10, 512)

        similarity, indices = self.engine.retrieve(query, gallery)

        self.assertEqual(similarity.shape, (5, 10))
        self.assertEqual(indices.shape, (5, 10))

    def test_retrieve_with_topk(self):
        """Test retrieve with top-k."""
        query = torch.randn(5, 512)
        gallery = torch.randn(10, 512)

        similarity, indices = self.engine.retrieve(query, gallery, top_k=3)

        self.assertEqual(indices.shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
