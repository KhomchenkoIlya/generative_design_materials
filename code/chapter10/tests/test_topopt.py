from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import (  # noqa: E402
    StructuredQuadMesh,
    cantilever_condition,
)
from project1_topopt.topopt import (  # noqa: E402
    SIMPConfig,
    build_density_filter,
    finite_difference_gradient_check,
    oc_update,
    resample_factor_two,
)


class TestDensityFilter(unittest.TestCase):
    def test_constant_field_is_preserved(self) -> None:
        mesh = StructuredQuadMesh(16, 6, length=3.0, height=1.0)
        filt = build_density_filter(mesh, 1.5)
        design = np.full(mesh.nelems, 0.4)
        physical = filt.apply(design)
        self.assertLess(float(np.max(np.abs(physical - 0.4))), 1e-14)

    def test_oc_update_respects_physical_volume(self) -> None:
        mesh = StructuredQuadMesh(12, 4, length=3.0, height=1.0)
        filt = build_density_filter(mesh, 1.5)

        design = np.full(mesh.nelems, 0.4)
        gradient = -np.linspace(1.0, 2.0, mesh.nelems)
        volume_gradient = filt.pullback(
            np.full(mesh.nelems, 1.0 / mesh.nelems)
        )

        updated = oc_update(
            design,
            gradient,
            volume_gradient,
            filt,
            volume_fraction=0.4,
            move=0.2,
        )
        physical = filt.apply(updated)

        self.assertLess(abs(float(np.mean(physical)) - 0.4), 1e-7)
        self.assertGreaterEqual(float(updated.min()), 0.0)
        self.assertLessEqual(float(updated.max()), 1.0)


class TestSensitivity(unittest.TestCase):
    def test_compliance_gradient_matches_finite_difference(self) -> None:
        mesh = StructuredQuadMesh(12, 4, length=3.0, height=1.0)
        condition = cantilever_condition(
            mesh,
            volume_fraction=0.4,
            load_y_fraction=0.5,
            load_angle_deg=-90.0,
        )
        config = SIMPConfig()

        check = finite_difference_gradient_check(
            mesh,
            condition,
            config,
            seed=17,
            count=6,
            step=1e-6,
        )

        self.assertLess(float(check["max_relative_error"]), 2e-5)


class TestResampling(unittest.TestCase):
    def test_factor_two_preserves_mean(self) -> None:
        nelx, nely = 8, 4
        density = np.linspace(0.1, 0.9, nelx * nely)

        coarse, cx, cy = resample_factor_two(
            density,
            nelx,
            nely,
            mode="coarsen",
        )
        fine, fx, fy = resample_factor_two(
            density,
            nelx,
            nely,
            mode="refine",
        )

        self.assertEqual((cx, cy), (4, 2))
        self.assertEqual((fx, fy), (16, 8))
        self.assertAlmostEqual(float(np.mean(coarse)), float(np.mean(density)))
        self.assertAlmostEqual(float(np.mean(fine)), float(np.mean(density)))


if __name__ == "__main__":
    unittest.main()
