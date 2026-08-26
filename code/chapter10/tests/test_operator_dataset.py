from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import StructuredQuadMesh  # noqa: E402
from project2_operator.dataset import (  # noqa: E402
    SNAPSHOT_FRACTIONS,
    build_conditions,
    condition_vector,
    element_to_node_density,
    snapshot_indices,
)


class TestConditionSplit(unittest.TestCase):
    def test_split_is_condition_level_and_has_expected_counts(self) -> None:
        conditions = build_conditions()

        self.assertEqual(len(conditions), 27)
        self.assertEqual(
            sum(item.split == "train" for item in conditions),
            18,
        )
        self.assertEqual(
            sum(item.split == "val" for item in conditions),
            4,
        )
        self.assertEqual(
            sum(item.split == "test" for item in conditions),
            5,
        )
        self.assertEqual(
            len({item.condition_id for item in conditions}),
            27,
        )

    def test_condition_vector_uses_direction_not_raw_angle_only(self) -> None:
        condition = build_conditions()[0]
        vector = condition_vector(condition)

        self.assertEqual(vector.shape, (4,))
        self.assertAlmostEqual(
            float(vector[2] ** 2 + vector[3] ** 2),
            1.0,
            places=6,
        )


class TestRepresentation(unittest.TestCase):
    def test_constant_element_density_stays_constant_on_nodes(self) -> None:
        mesh = StructuredQuadMesh(
            nelx=8,
            nely=4,
            length=3.0,
            height=1.0,
        )
        density = np.full(mesh.nelems, 0.4)

        nodal = element_to_node_density(density, mesh)

        self.assertEqual(nodal.shape, (5, 9))
        self.assertLess(
            float(np.max(np.abs(nodal - 0.4))),
            1e-7,
        )

    def test_six_snapshot_indices_include_endpoints(self) -> None:
        indices = snapshot_indices(101)

        self.assertEqual(len(indices), len(SNAPSHOT_FRACTIONS))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 100)
        self.assertEqual(len(set(indices)), len(indices))


if __name__ == "__main__":
    unittest.main()
