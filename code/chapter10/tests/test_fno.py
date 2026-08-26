from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from project2_operator.fno import (  # noqa: E402
    FNO2d,
    build_raw_inputs,
    channel_statistics,
    coordinate_grid,
    output_rms,
)


class TestFNOInput(unittest.TestCase):
    def test_input_shape_is_nine_channels(self) -> None:
        n, h, w = 3, 33, 97
        rho = np.full((n, h, w), 0.4, dtype=np.float32)
        force = np.zeros((n, 2, h, w), dtype=np.float32)
        condition = np.array(
            [
                [0.4, 0.5, 0.0, -1.0],
                [0.3, 0.25, 0.2, -0.98],
                [0.5, 0.75, -0.2, -0.98],
            ],
            dtype=np.float32,
        )

        raw = build_raw_inputs(rho, force, condition)

        self.assertEqual(raw.shape, (n, 9, h, w))
        self.assertTrue(np.all(np.isfinite(raw)))

    def test_statistics_use_train_subset(self) -> None:
        raw = np.zeros((4, 9, 5, 7), dtype=np.float32)
        raw[0:2] = 1.0
        raw[2:4] = 100.0
        mask = np.array([True, True, False, False])

        mean, std = channel_statistics(raw, mask)

        self.assertLess(float(np.max(np.abs(mean - 1.0))), 1e-7)
        self.assertTrue(np.all(std >= 1e-6))

    def test_coordinate_grid_has_expected_endpoints(self) -> None:
        grid = coordinate_grid(5, 7)

        self.assertEqual(grid.shape, (2, 5, 7))
        self.assertAlmostEqual(float(grid[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(grid[0, 0, -1]), 1.0)
        self.assertAlmostEqual(float(grid[1, 0, 0]), 0.0)
        self.assertAlmostEqual(float(grid[1, -1, 0]), 1.0)


class TestFNOModel(unittest.TestCase):
    def test_model_runs_on_training_and_finer_resolution(self) -> None:
        torch.manual_seed(1)
        model = FNO2d(
            width=8,
            modes_y=4,
            modes_x=6,
            layers=2,
            padding_y=2,
            padding_x=2,
        )

        for h, w in ((33, 97), (65, 193)):
            x = torch.randn(2, 9, h, w)
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 2, h, w))
            self.assertTrue(torch.isfinite(y).all())
            self.assertEqual(
                float(torch.max(torch.abs(y[:, :, :, 0])).detach()),
                0.0,
            )

    def test_output_rms_is_positive(self) -> None:
        displacement = np.zeros(
            (4, 2, 5, 7),
            dtype=np.float32,
        )
        displacement[:2, 0] = 2.0
        displacement[:2, 1] = 3.0
        mask = np.array([True, True, False, False])

        scale = output_rms(displacement, mask)

        self.assertEqual(scale.shape, (1, 2, 1, 1))
        self.assertAlmostEqual(float(scale[0, 0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(scale[0, 1, 0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
