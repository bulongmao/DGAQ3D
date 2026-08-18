#!/usr/bin/env python
"""Focused tests for the paired qualitative export primitives."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from export_paired_qualitative import (
    global_to_lidar,
    projected_segments,
    save_pdf_pages,
    select_candidates,
)


class PairedQualitativeExportTest(unittest.TestCase):

    def test_global_to_lidar_identity(self):
        raw_info = {
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'lidar2ego_translation': [0.0, 0.0, 0.0],
        }
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
        np.testing.assert_allclose(global_to_lidar(points, raw_info), points)

    def test_near_plane_segment_projection(self):
        corners = np.array([
            [-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1],
            [-1, -1, 5], [-1, 1, 5], [1, -1, 5], [1, 1, 5],
        ], dtype=np.float64)
        projection = np.array([
            [80, 0, 50, 0],
            [0, 80, 50, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        segments = projected_segments(
            corners, projection,
            ((0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 7)),
            (100, 100), 0.1)
        self.assertGreater(len(segments), 0)
        for segment in segments:
            for x, y in segment:
                self.assertGreaterEqual(x, 0)
                self.assertLessEqual(x, 99)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(y, 99)

    def test_selection_reserves_regression_and_uses_distinct_cameras(self):
        def row(index, status, challenge, class_name, scene, camera, gain):
            return {
                "sample_index": index,
                "status": status,
                "challenge": challenge,
                "class_name": class_name,
                "scene_token": scene,
                "camera_name": camera,
                "error_gain_m": gain,
                "relative_distance": 0.8,
                "projected_size_px": 20.0,
            }

        rows = [
            row(0, "ours_only", "far_small", "car", "a",
                "CAM_FRONT", None),
            row(0, "localization_gain", "far", "truck", "a",
                "CAM_BACK", 1.2),
            row(1, "localization_gain", "far", "truck", "b",
                "CAM_FRONT_LEFT", 0.8),
            row(2, "modest_gain", "small", "pedestrian", "c",
                "CAM_BACK_LEFT", 0.2),
            row(3, "baseline_only", "far_small", "bus", "d",
                "CAM_BACK_RIGHT", None),
        ]
        args = SimpleNamespace(num_cases=4, num_regressions=1)
        selected = select_candidates(rows, args)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({item["sample_index"] for item in selected}), 4)
        self.assertEqual(len({item["camera_name"] for item in selected}), 4)
        self.assertEqual(
            sum(item["status"] == "baseline_only" for item in selected), 1)

    def test_png_and_multipage_pdf_export(self):
        rows = [
            Image.new('RGB', (320, 120), (255, 255, 255)),
            Image.new('RGB', (320, 120), (245, 245, 245)),
        ]
        args = SimpleNamespace(rows_per_page=1)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            pdf_path, png_path, pages = save_pdf_pages(rows, output_dir, args)
            self.assertEqual(len(pages), 2)
            self.assertTrue(pdf_path.is_file())
            self.assertTrue(png_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 100)
            self.assertGreater(png_path.stat().st_size, 100)
            with open(pdf_path, 'rb') as stream:
                self.assertEqual(stream.read(4), b'%PDF')


if __name__ == '__main__':
    unittest.main()
