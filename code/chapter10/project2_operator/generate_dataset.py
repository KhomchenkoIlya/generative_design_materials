"""CLI для воспроизводимой генерации V04 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from project2_operator.dataset import generate_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    manifest = generate_dataset(
        args.dataset,
        args.manifest,
        args.conditions,
        args.figure,
    )

    print("V04 operator dataset")
    print(
        "conditions="
        f"{manifest['condition_grid']['number_of_conditions']}, "
        "samples="
        f"{manifest['snapshots']['total_samples']}"
    )
    print(
        "condition split="
        f"{manifest['split']['condition_counts']}"
    )
    print(
        "sample split="
        f"{manifest['split']['sample_counts']}"
    )
    print(
        "iterations="
        f"{manifest['quality']['iteration_min']}.."
        f"{manifest['quality']['iteration_max']}"
    )
    print(
        "max FEM residual="
        f"{manifest['quality']['max_relative_residual']:.3e}"
    )
    print(
        "dataset bytes="
        f"{manifest['dataset_bytes']}, "
        "sha256="
        f"{manifest['dataset_sha256']}"
    )
    print(
        "runtime seconds="
        f"{manifest['runtime_seconds']:.1f}"
    )


if __name__ == "__main__":
    main()
