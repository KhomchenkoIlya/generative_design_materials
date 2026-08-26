from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from project2_operator.analyze_fno import (  # noqa: E402
    pairwise_order_accuracy,
    spearman_no_ties,
)


class TestRankingMetrics(unittest.TestCase):
    def test_pairwise_order_accuracy_perfect(self) -> None:
        exact = np.array([10.0, 7.0, 4.0, 2.0])
        predicted = np.array([9.0, 8.0, 3.0, 1.0])

        self.assertAlmostEqual(
            pairwise_order_accuracy(exact, predicted),
            1.0,
        )

    def test_pairwise_order_accuracy_reversed(self) -> None:
        exact = np.array([1.0, 2.0, 3.0])
        predicted = np.array([3.0, 2.0, 1.0])

        self.assertAlmostEqual(
            pairwise_order_accuracy(exact, predicted),
            0.0,
        )

    def test_spearman_no_ties(self) -> None:
        exact = np.array([1.0, 3.0, 2.0, 4.0])

        self.assertAlmostEqual(
            spearman_no_ties(exact, exact),
            1.0,
        )
        self.assertAlmostEqual(
            spearman_no_ties(exact, -exact),
            -1.0,
        )


if __name__ == "__main__":
    unittest.main()
