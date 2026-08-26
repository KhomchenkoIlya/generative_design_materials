from __future__ import annotations

from pathlib import Path
import sys
import unittest

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from summarize_iteration10 import (  # noqa: E402
    diffusion_role,
    fno_role,
    percentage,
)


class TestIteration10Summary(unittest.TestCase):
    def test_percentage(self) -> None:
        self.assertAlmostEqual(
            percentage(0.9466666667),
            94.66666667,
        )

    def test_fno_screening_role(self) -> None:
        self.assertEqual(
            fno_role(
                speedup=5.27,
                pairwise_ordering=0.947,
                physics_residual=64.5,
            ),
            "screening_surrogate_not_final_solver",
        )

    def test_diffusion_candidate_role(self) -> None:
        self.assertEqual(
            diffusion_role(
                connected_fraction=6 / 80,
                warmstart_ratio=0.99798,
                rms_change=0.40449,
            ),
            "candidate_generator_with_modest_warmstart_effect",
        )


if __name__ == "__main__":
    unittest.main()
