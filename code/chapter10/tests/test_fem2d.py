from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse.linalg import eigsh

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import (  # noqa: E402
    StructuredQuadMesh,
    assemble_stiffness,
    cantilever_condition,
    half_mbb_condition,
    quad4_stiffness,
    solve_elasticity,
    sparse_symmetry_error,
)


class TestElement(unittest.TestCase):
    def test_q4_has_three_rigid_modes(self) -> None:
        mesh = StructuredQuadMesh(1, 1, length=1.0, height=1.0)
        coords = mesh.coordinates()[mesh.elements()[0]]
        ke = quad4_stiffness(coords, young=1.0, poisson=0.3)

        self.assertLess(float(np.max(np.abs(ke - ke.T))), 1e-12)

        eigenvalues = np.linalg.eigvalsh(ke)
        self.assertEqual(int(np.count_nonzero(np.abs(eigenvalues) < 1e-10)), 3)
        self.assertGreater(float(eigenvalues[3]), 1e-6)


class TestGlobalProblems(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mesh = StructuredQuadMesh(12, 4, length=3.0, height=1.0)
        cls.stiffness = assemble_stiffness(
            cls.mesh,
            young=1.0,
            poisson=0.3,
        )

    def _check_solution(self, condition) -> None:
        result = solve_elasticity(self.stiffness, condition)

        self.assertLess(sparse_symmetry_error(self.stiffness), 1e-13)
        self.assertLess(result.relative_residual, 1e-10)
        self.assertLess(result.energy_identity_error, 1e-10)
        self.assertLess(result.force_balance_error, 1e-9)
        self.assertGreater(result.compliance, 0.0)

        self.assertLess(
            float(np.max(np.abs(result.displacement[condition.fixed_dofs]))),
            1e-14,
        )

        kff = self.stiffness[result.free_dofs][:, result.free_dofs]
        smallest = float(eigsh(kff, k=1, which="SA", return_eigenvectors=False)[0])
        self.assertGreater(smallest, 0.0)

    def test_cantilever(self) -> None:
        condition = cantilever_condition(
            self.mesh,
            volume_fraction=0.4,
            load_y_fraction=0.5,
            load_angle_deg=-90.0,
        )
        self._check_solution(condition)

        left = self.mesh.left_nodes()
        expected_fixed = 2 * left.size
        self.assertEqual(condition.fixed_dofs.size, expected_fixed)
        self.assertAlmostEqual(float(np.linalg.norm(condition.force)), 1.0)

    def test_half_mbb(self) -> None:
        condition = half_mbb_condition(
            self.mesh,
            volume_fraction=0.4,
        )
        self._check_solution(condition)

        expected_fixed = self.mesh.nely + 2
        self.assertEqual(condition.fixed_dofs.size, expected_fixed)
        self.assertAlmostEqual(float(np.linalg.norm(condition.force)), 1.0)

    def test_cantilever_condition_is_parameterized(self) -> None:
        lower = cantilever_condition(
            self.mesh,
            volume_fraction=0.35,
            load_y_fraction=0.25,
            load_angle_deg=-45.0,
        )
        upper = cantilever_condition(
            self.mesh,
            volume_fraction=0.50,
            load_y_fraction=0.75,
            load_angle_deg=-135.0,
        )

        self.assertAlmostEqual(lower.volume_fraction, 0.35)
        self.assertAlmostEqual(upper.volume_fraction, 0.50)
        self.assertFalse(np.array_equal(lower.force, upper.force))


if __name__ == "__main__":
    unittest.main()
