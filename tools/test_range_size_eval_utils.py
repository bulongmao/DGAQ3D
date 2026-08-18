#!/usr/bin/env python
"""Synthetic regression tests for range/size evaluation primitives."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_range_size_breakdown as integration
from range_size_eval_utils import (
    apply_image_transform,
    assign_bin,
    build_test_ida,
    match_predictions,
    max_projected_box_size,
    max_projected_box_sizes,
    nuscenes_ap_from_tp,
    quaternion_to_matrix,
    recall_from_tp,
    repeat_bootstrap_events,
)


class RangeSizeEvalUtilsTest(unittest.TestCase):

    def test_bin_boundaries_are_half_open_and_last_is_closed(self):
        edges = [0.0, 20.0, 30.0, 50.0]
        values = [-1.0, 0.0, 19.999, 20.0, 30.0, 50.0, 50.1, np.nan]
        expected = [-1, 0, 0, 1, 2, 2, -1, -1]
        np.testing.assert_array_equal(assign_bin(values, edges), expected)

    def test_keyframe_validation_ida_matches_repository_pipeline(self):
        geometry = build_test_ida({
            'H': 900,
            'W': 1600,
            'final_dim': (640, 1600),
            'resize_test': 0.04,
            'crop_h': (0.0, 0.0),
        }, transform_type='ResizeCropFlipImageV2')
        self.assertAlmostEqual(geometry['resize'], 1.04)
        self.assertEqual(geometry['resize_dims'], (1664, 936))
        self.assertEqual(geometry['crop'], (32, 296, 1632, 936))
        self.assertEqual(geometry['final_dim'], (640, 1600))
        point = np.array([100.0, 400.0, 1.0, 1.0])
        transformed = geometry['matrix'] @ point
        np.testing.assert_allclose(transformed[:2], [72.0, 120.0])

    def test_quaternion_and_lidar2image_composition(self):
        np.testing.assert_allclose(
            quaternion_to_matrix([1.0, 0.0, 0.0, 0.0]), np.eye(3))
        matrix = np.eye(4)
        ida = np.eye(4)
        ida[0, 0] = ida[1, 1] = 2.0
        ida[0, 2] = -5.0
        transformed = apply_image_transform(matrix, ida)
        np.testing.assert_allclose(transformed, ida)

    @staticmethod
    def cube_corners():
        # MMDetection3D corner order: 0, 1, 3, 2 on each z plane.
        return np.array([
            [-1.0, -1.0, 9.0],
            [-1.0, 1.0, 9.0],
            [1.0, -1.0, 9.0],
            [1.0, 1.0, 9.0],
            [-1.0, -1.0, 11.0],
            [-1.0, 1.0, 11.0],
            [1.0, -1.0, 11.0],
            [1.0, 1.0, 11.0],
        ], dtype=np.float64)

    def test_projected_size_and_vectorized_path_agree(self):
        camera = np.array([
            [100.0, 0.0, 50.0, 0.0],
            [0.0, 100.0, 50.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        corners = self.cube_corners()
        scalar_size, scalar_views = max_projected_box_size(
            corners, [camera], (100, 100))
        batch_size, batch_views = max_projected_box_sizes(
            np.stack([corners, corners]), [camera], (100, 100), chunk_size=1)
        self.assertAlmostEqual(scalar_size, 200.0 / 9.0, places=5)
        self.assertEqual(scalar_views, 1)
        np.testing.assert_allclose(batch_size, [scalar_size, scalar_size])
        np.testing.assert_array_equal(batch_views, [1, 1])

    def test_score_ordered_nearest_matching_rejects_duplicate(self):
        gt = np.array([[0.0, 0.0]])
        predictions = np.array([[0.2, 0.0], [0.1, 0.0]])
        scores = np.array([0.8, 0.9])
        order, matched = match_predictions(gt, predictions, scores, 0.5)
        np.testing.assert_array_equal(order, [1, 0])
        np.testing.assert_array_equal(matched, [0, -1])

    def test_nuscenes_ap_and_recall_endpoints(self):
        self.assertAlmostEqual(nuscenes_ap_from_tp([1], 1), 1.0)
        self.assertAlmostEqual(recall_from_tp([1, 0], 2), 0.5)
        self.assertEqual(nuscenes_ap_from_tp([], 1), 0.0)
        self.assertEqual(nuscenes_ap_from_tp([0, 0], 2), 0.0)

    def test_cluster_bootstrap_duplicates_complete_scene_events(self):
        sampled_tp, sampled_gt = repeat_bootstrap_events(
            tp_sorted=[1, 0, 1],
            event_units_sorted=[0, 1, 0],
            gt_units=[0, 1],
            unit_counts=[2, 0])
        np.testing.assert_array_equal(sampled_tp, [1, 1, 1, 1])
        self.assertEqual(sampled_gt, 2)

    def test_bucket_matching_ignores_other_bucket_true_positive(self):
        gt_class = {
            'unit': np.array([0, 0], dtype=np.int32),
            'absolute': np.array([0, 1], dtype=np.int16),
        }
        # First event matches bucket 1. It must not become an FP in bucket 0.
        event_class = {
            'tp': np.array([1, 0], dtype=np.int8),
            'unit': np.array([0, 0], dtype=np.int32),
            'absolute': np.array([1, 0], dtype=np.int16),
        }
        ap_near, recall_near, gt_near = integration.class_metric(
            gt_class, event_class, 'absolute', 0, None, 0.1, 0.1)
        ap_far, recall_far, gt_far = integration.class_metric(
            gt_class, event_class, 'absolute', 1, None, 0.1, 0.1)
        self.assertEqual((gt_near, ap_near, recall_near), (1, 0.0, 0.0))
        self.assertEqual(gt_far, 1)
        self.assertAlmostEqual(ap_far, 1.0)
        self.assertAlmostEqual(recall_far, 1.0)


if __name__ == '__main__':
    unittest.main()
