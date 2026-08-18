#!/usr/bin/env python
"""Tests for the compact paired-qualitative paper layout."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from paper_qualitative_layout import (COLORS, build_main_figure,
                                      make_labeled_panel)


class PaperQualitativeLayoutTest(unittest.TestCase):

    def test_panel_label_is_above_image(self):
        source_color = (21, 42, 63)
        source = Image.new('RGB', (80, 40), source_color)
        panel = make_labeled_panel(
            source, (80, 40), '3DPPE | miss', COLORS['baseline'])
        self.assertEqual(panel.size, (80, 108))
        self.assertEqual(panel.getpixel((79, 0)), COLORS['paper'])
        self.assertEqual(panel.getpixel((40, 88)), source_color)

    def test_builds_three_case_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_dir = root / 'cases'
            cases_dir.mkdir()
            metadata = []
            colors = ((240, 235, 225), (225, 240, 245),
                      (225, 245, 230), (240, 240, 240))
            for rank in (1, 4, 6):
                row = Image.new('RGB', (2104, 294), 'white')
                for index, color in enumerate(colors):
                    row.paste(Image.new('RGB', (520, 208), color),
                              (index * 528, 86))
                row.save(cases_dir / '{:02d}_case.png'.format(rank))
                metadata.append({
                    'selection_rank': rank,
                    'class_name': 'pedestrian',
                    'distance_m': 35.0 + rank,
                    'projected_size_px': 24.0,
                    'status': 'baseline_only' if rank == 6 else 'ours_only',
                    'baseline_error_m': 0.4 if rank == 6 else None,
                    'ours_error_m': None if rank == 6 else 0.2,
                    'target_rect': [700.0, 200.0, 730.0, 245.0],
                })
            with open(root / 'selected_cases.json', 'w') as stream:
                json.dump(metadata, stream)
            png_path, pdf_path = build_main_figure(root)
            self.assertTrue(png_path.is_file())
            self.assertTrue(pdf_path.is_file())
            with Image.open(png_path) as figure:
                self.assertGreater(figure.width, figure.height)
                self.assertGreater(figure.width, 2000)
            with open(pdf_path, 'rb') as stream:
                self.assertEqual(stream.read(4), b'%PDF')


if __name__ == '__main__':
    unittest.main()
