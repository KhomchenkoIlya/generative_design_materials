from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from project3_diffusion.diffusion import (  # noqa: E402
    ConditionalUNet,
    DiffusionSchedule,
    condition_matrix,
    downsample_density_2x,
    volume_project,
)


class TestDensityPreparation(unittest.TestCase):
    def test_downsample_preserves_mean(self) -> None:
        rng = np.random.default_rng(1)
        rho = rng.uniform(
            0.0,
            1.0,
            size=(3, 32, 96),
        ).astype(np.float32)

        small = downsample_density_2x(rho)

        self.assertEqual(
            small.shape,
            (3, 16, 48),
        )
        self.assertLess(
            abs(float(rho.mean() - small.mean())),
            1e-7,
        )

    def test_volume_projection_hits_target(self) -> None:
        rng = np.random.default_rng(2)
        rho = rng.uniform(
            0.0,
            1.0,
            size=(16, 48),
        )

        projected = volume_project(
            rho,
            0.37,
        )

        self.assertTrue(
            np.all(
                (projected >= 0.0)
                & (projected <= 1.0)
            )
        )
        self.assertLess(
            abs(float(projected.mean()) - 0.37),
            2e-7,
        )

    def test_condition_matrix_has_progress(self) -> None:
        condition = np.array(
            [
                [0.4, 0.5, 0.0, -1.0],
                [0.3, 0.25, 0.2, -0.98],
            ],
            dtype=np.float32,
        )
        progress = np.array(
            [0.0, 1.0],
            dtype=np.float32,
        )

        matrix = condition_matrix(
            condition,
            progress,
        )

        self.assertEqual(matrix.shape, (2, 5))
        self.assertAlmostEqual(
            float(matrix[1, -1]),
            1.0,
        )


class TestDiffusionModel(unittest.TestCase):
    def test_unet_and_schedule_shapes(self) -> None:
        torch.manual_seed(3)

        model = ConditionalUNet(
            base_channels=16,
            embedding_dim=32,
            condition_dim=5,
        )
        schedule = DiffusionSchedule(
            steps=10,
        )

        x0 = torch.randn(2, 1, 16, 48)
        condition = torch.randn(2, 5)
        timestep = torch.tensor(
            [0, 9],
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)

        xt = schedule.q_sample(
            x0,
            timestep,
            noise,
        )
        prediction = model(
            xt,
            timestep,
            condition,
        )

        self.assertEqual(
            tuple(prediction.shape),
            (2, 1, 16, 48),
        )
        self.assertTrue(
            torch.isfinite(prediction).all()
        )


if __name__ == "__main__":
    unittest.main()
