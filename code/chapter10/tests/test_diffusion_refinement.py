from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import StructuredQuadMesh, cantilever_condition
from project1_topopt.topopt import SIMPConfig
from project3_diffusion.evaluate_refinement import (
    design_to_field,
    field_to_design,
    fixed_budget_refinement,
    support_to_load_path,
    upsample_density_2x,
)


class TestV08Representation(unittest.TestCase):
    def test_upsample_preserves_mean(self) -> None:
        rng = np.random.default_rng(8)
        coarse = rng.uniform(0.0, 1.0, size=(16, 48))
        fine = upsample_density_2x(coarse)
        self.assertEqual(fine.shape, (32, 96))
        self.assertAlmostEqual(float(fine.mean()), float(coarse.mean()), places=12)

    def test_field_design_round_trip(self) -> None:
        mesh = StructuredQuadMesh(nelx=6, nely=2, length=3.0, height=1.0)
        field = np.arange(12, dtype=float).reshape(2, 6) / 11.0
        design = field_to_design(field, mesh)
        recovered = design_to_field(design, mesh)
        np.testing.assert_allclose(recovered, field)


class TestV08Connectivity(unittest.TestCase):
    def test_support_to_load_path(self) -> None:
        field = np.zeros((4, 8), dtype=float)
        field[2, :] = 1.0
        self.assertTrue(support_to_load_path(field, 0.5))

        field[2, 4] = 0.0
        self.assertFalse(support_to_load_path(field, 0.5))


class TestV08Refinement(unittest.TestCase):
    def test_fixed_budget_refinement_runs_exact_steps(self) -> None:
        mesh = StructuredQuadMesh(nelx=12, nely=4, length=3.0, height=1.0)
        problem = cantilever_condition(
            mesh,
            volume_fraction=0.4,
            load_y_fraction=0.5,
            load_angle_deg=-90.0,
        )
        config = SIMPConfig(
            penal=3.0,
            rmin=1.5,
            emin_ratio=1.0e-9,
            move=0.2,
            tolerance=1.0e-2,
            max_iterations=2,
        )
        initial = np.full((mesh.nely, mesh.nelx), 0.4, dtype=float)
        result = fixed_budget_refinement(
            mesh,
            problem,
            initial,
            config,
            steps=2,
        )

        self.assertEqual(len(result["history"]), 2)
        self.assertEqual(result["field"].shape, (mesh.nely, mesh.nelx))
        self.assertLess(abs(float(result["volume_fraction"]) - 0.4), 1e-6)
        self.assertLess(float(result["relative_residual"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
