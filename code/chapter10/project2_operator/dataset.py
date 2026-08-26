"""Генерация FEM/SIMP-датасета для проекта 2.

Каждое условие — отдельная cantilever-задача:
    target volume fraction,
    положение единичной силы на правой границе,
    направление силы.

Для условия строится полная SIMP-траектория. Из неё берутся шесть состояний:
начальное, четыре промежуточных и терминальное. Split выполняется по condition_id,
поэтому состояния одной траектории никогда не попадают в разные выборки.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib

import numpy as np

from common.fem2d import StructuredQuadMesh, cantilever_condition
from project1_topopt.topopt import (
    SIMPConfig,
    analyze_design,
    build_density_filter,
    oc_update,
)


SNAPSHOT_FRACTIONS = (0.0, 0.05, 0.15, 0.35, 0.65, 1.0)
SPLIT_SEED = 20260813
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class DatasetCondition:
    condition_id: int
    split: str
    target_volume: float
    load_y_fraction: float
    load_angle_deg: float


@dataclass(frozen=True)
class TrajectoryState:
    iteration: int
    physical_density: np.ndarray
    displacement: np.ndarray
    compliance: float
    relative_residual: float


@dataclass(frozen=True)
class TrajectoryResult:
    states: tuple[TrajectoryState, ...]
    iterations: int
    converged: bool
    final_change: float


def build_conditions() -> tuple[DatasetCondition, ...]:
    """Полный 3x3x3 grid и детерминированный split 18/4/5."""

    raw: list[tuple[float, float, float]] = []
    for volume in (0.30, 0.40, 0.50):
        for load_y in (0.25, 0.50, 0.75):
            for angle in (-105.0, -90.0, -75.0):
                raw.append((volume, load_y, angle))

    rng = np.random.default_rng(SPLIT_SEED)
    permutation = rng.permutation(len(raw))

    split_by_index: dict[int, str] = {}
    for index in permutation[:18]:
        split_by_index[int(index)] = "train"
    for index in permutation[18:22]:
        split_by_index[int(index)] = "val"
    for index in permutation[22:]:
        split_by_index[int(index)] = "test"

    conditions = tuple(
        DatasetCondition(
            condition_id=index,
            split=split_by_index[index],
            target_volume=volume,
            load_y_fraction=load_y,
            load_angle_deg=angle,
        )
        for index, (volume, load_y, angle) in enumerate(raw)
    )

    return conditions


def condition_vector(condition: DatasetCondition) -> np.ndarray:
    """Компактное условие: объём, положение, cos(theta), sin(theta)."""

    angle = math.radians(condition.load_angle_deg)
    return np.array(
        [
            condition.target_volume,
            condition.load_y_fraction,
            math.cos(angle),
            math.sin(angle),
        ],
        dtype=np.float32,
    )


def element_to_node_density(
    element_density: np.ndarray,
    mesh: StructuredQuadMesh,
) -> np.ndarray:
    """Усреднить физическую плотность соседних элементов в узлы."""

    field = np.asarray(element_density, dtype=float).reshape(
        mesh.nelx,
        mesh.nely,
    )

    accumulation = np.zeros((mesh.nelx + 1, mesh.nely + 1), dtype=float)
    counts = np.zeros_like(accumulation)

    for dx in (0, 1):
        for dy in (0, 1):
            accumulation[
                dx : dx + mesh.nelx,
                dy : dy + mesh.nely,
            ] += field
            counts[
                dx : dx + mesh.nelx,
                dy : dy + mesh.nely,
            ] += 1.0

    nodal = accumulation / counts
    return nodal.T.astype(np.float32)


def displacement_to_grid(
    displacement: np.ndarray,
    mesh: StructuredQuadMesh,
) -> np.ndarray:
    """Вектор FEM -> два nodal-канала формы (2, nely+1, nelx+1)."""

    nodes = np.asarray(displacement, dtype=float).reshape(mesh.nnodes, 2)
    grid = nodes.reshape(mesh.nelx + 1, mesh.nely + 1, 2)
    grid = np.transpose(grid, (2, 1, 0))
    return grid.astype(np.float32)


def force_to_grid(
    force: np.ndarray,
    mesh: StructuredQuadMesh,
) -> np.ndarray:
    return displacement_to_grid(force, mesh)


def fixed_mask(mesh: StructuredQuadMesh) -> np.ndarray:
    """Два бинарных канала для закреплённых nodal DOF консоли."""

    mask = np.zeros((2, mesh.nely + 1, mesh.nelx + 1), dtype=np.float32)
    mask[:, :, 0] = 1.0
    return mask


def optimize_trajectory(
    mesh: StructuredQuadMesh,
    condition: DatasetCondition,
    config: SIMPConfig,
) -> TrajectoryResult:
    """SIMP-траектория с одним FEM-анализом на состояние."""

    problem = cantilever_condition(
        mesh,
        volume_fraction=condition.target_volume,
        load_y_fraction=condition.load_y_fraction,
        load_angle_deg=condition.load_angle_deg,
        load_magnitude=1.0,
    )
    density_filter = build_density_filter(mesh, config.rmin)

    design = np.full(
        mesh.nelems,
        condition.target_volume,
        dtype=float,
    )

    analysis = analyze_design(
        mesh,
        problem,
        design,
        density_filter,
        config,
    )

    states: list[TrajectoryState] = [
        TrajectoryState(
            iteration=0,
            physical_density=analysis.physical_density.astype(
                np.float32,
                copy=True,
            ),
            displacement=analysis.displacement.astype(
                np.float32,
                copy=True,
            ),
            compliance=analysis.compliance,
            relative_residual=analysis.relative_residual,
        )
    ]

    converged = False
    final_change = float("inf")

    for iteration in range(1, config.max_iterations + 1):
        updated = oc_update(
            design,
            analysis.gradient,
            analysis.volume_gradient,
            density_filter,
            condition.target_volume,
            config.move,
        )
        final_change = float(np.max(np.abs(updated - design)))
        design = updated

        analysis = analyze_design(
            mesh,
            problem,
            design,
            density_filter,
            config,
        )

        states.append(
            TrajectoryState(
                iteration=iteration,
                physical_density=analysis.physical_density.astype(
                    np.float32,
                    copy=True,
                ),
                displacement=analysis.displacement.astype(
                    np.float32,
                    copy=True,
                ),
                compliance=analysis.compliance,
                relative_residual=analysis.relative_residual,
            )
        )

        if final_change < config.tolerance:
            converged = True
            break

    return TrajectoryResult(
        states=tuple(states),
        iterations=len(states) - 1,
        converged=converged,
        final_change=final_change,
    )


def snapshot_indices(number_of_states: int) -> tuple[int, ...]:
    if number_of_states < len(SNAPSHOT_FRACTIONS):
        raise ValueError("Траектория слишком короткая для шести снимков.")

    last = number_of_states - 1
    indices = tuple(
        int(round(fraction * last))
        for fraction in SNAPSHOT_FRACTIONS
    )

    if len(set(indices)) != len(indices):
        raise ValueError("Snapshot indices должны быть различными.")

    return indices


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_density_montage(
    densities: list[np.ndarray],
    mesh: StructuredQuadMesh,
    path: Path,
    *,
    scale: int = 4,
    gap: int = 4,
) -> None:
    """Горизонтальная PNG-полоска шести плотностей; чёрное = материал."""

    images: list[np.ndarray] = []

    for density in densities:
        field = np.asarray(density, dtype=float).reshape(
            mesh.nelx,
            mesh.nely,
        ).T
        field = np.flipud(field)
        gray = np.rint(
            255.0 * (1.0 - np.clip(field, 0.0, 1.0))
        ).astype(np.uint8)
        image = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)
        images.append(image)

    height = images[0].shape[0]
    separator = np.full((height, gap), 255, dtype=np.uint8)

    canvas_parts: list[np.ndarray] = []
    for index, image in enumerate(images):
        if index:
            canvas_parts.append(separator)
        canvas_parts.append(image)

    canvas = np.concatenate(canvas_parts, axis=1)
    png_height, png_width = canvas.shape

    raw = b"".join(
        b"\x00" + canvas[row].tobytes()
        for row in range(png_height)
    )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                png_width,
                png_height,
                8,
                0,
                0,
                0,
                0,
            ),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(
    dataset_path: Path,
    manifest_path: Path,
    conditions_path: Path,
    figure_path: Path,
) -> dict[str, object]:
    """Полностью построить локальный NPZ и маленькие tracked-метаданные."""

    mesh = StructuredQuadMesh(
        nelx=96,
        nely=32,
        length=3.0,
        height=1.0,
    )
    config = SIMPConfig(
        penal=3.0,
        rmin=1.5,
        emin_ratio=1.0e-9,
        move=0.2,
        tolerance=1.0e-2,
        max_iterations=800,
    )

    conditions = build_conditions()
    condition_by_id = {item.condition_id: item for item in conditions}

    rho_element_samples: list[np.ndarray] = []
    rho_node_samples: list[np.ndarray] = []
    force_samples: list[np.ndarray] = []
    displacement_samples: list[np.ndarray] = []
    condition_samples: list[np.ndarray] = []
    compliance_samples: list[float] = []
    residual_samples: list[float] = []
    condition_id_samples: list[int] = []
    split_samples: list[int] = []
    iteration_samples: list[int] = []
    progress_samples: list[float] = []
    snapshot_rank_samples: list[int] = []

    condition_rows: list[dict[str, object]] = []
    montage_densities: list[np.ndarray] | None = None

    start_all = time.perf_counter()

    for ordinal, condition in enumerate(conditions, start=1):
        print(
            f"[{ordinal:02d}/27] condition={condition.condition_id:02d} "
            f"split={condition.split:5s} "
            f"V={condition.target_volume:.2f} "
            f"y={condition.load_y_fraction:.2f} "
            f"angle={condition.load_angle_deg:+.0f}"
        )

        start = time.perf_counter()
        trajectory = optimize_trajectory(mesh, condition, config)
        runtime = time.perf_counter() - start

        if not trajectory.converged:
            raise RuntimeError(
                f"condition {condition.condition_id} не сошлось "
                f"за {config.max_iterations} итераций."
            )

        indices = snapshot_indices(len(trajectory.states))
        problem = cantilever_condition(
            mesh,
            volume_fraction=condition.target_volume,
            load_y_fraction=condition.load_y_fraction,
            load_angle_deg=condition.load_angle_deg,
            load_magnitude=1.0,
        )
        force_grid = force_to_grid(problem.force, mesh)

        selected_densities: list[np.ndarray] = []

        for rank, state_index in enumerate(indices):
            state = trajectory.states[state_index]
            selected_densities.append(state.physical_density)

            rho_element_samples.append(
                state.physical_density.reshape(
                    mesh.nelx,
                    mesh.nely,
                ).T.astype(np.float32)
            )
            rho_node_samples.append(
                element_to_node_density(
                    state.physical_density,
                    mesh,
                )
            )
            force_samples.append(force_grid)
            displacement_samples.append(
                displacement_to_grid(
                    state.displacement,
                    mesh,
                )
            )
            condition_samples.append(condition_vector(condition))
            compliance_samples.append(state.compliance)
            residual_samples.append(state.relative_residual)
            condition_id_samples.append(condition.condition_id)
            split_samples.append(SPLIT_CODE[condition.split])
            iteration_samples.append(state.iteration)
            progress_samples.append(
                state_index / (len(trajectory.states) - 1)
            )
            snapshot_rank_samples.append(rank)

        if (
            condition.target_volume == 0.40
            and condition.load_y_fraction == 0.50
            and condition.load_angle_deg == -90.0
        ):
            montage_densities = selected_densities

        condition_rows.append(
            {
                "condition_id": condition.condition_id,
                "split": condition.split,
                "target_volume": condition.target_volume,
                "load_y_fraction": condition.load_y_fraction,
                "load_angle_deg": condition.load_angle_deg,
                "iterations": trajectory.iterations,
                "converged": int(trajectory.converged),
                "final_change": trajectory.final_change,
                "terminal_compliance": trajectory.states[-1].compliance,
                "runtime_seconds": runtime,
            }
        )

    if montage_densities is None:
        raise RuntimeError("Не найдено базовое условие для montage.")

    arrays = {
        "rho_element": np.stack(rho_element_samples).astype(np.float32),
        "rho_node": np.stack(rho_node_samples).astype(np.float32),
        "force": np.stack(force_samples).astype(np.float32),
        "displacement": np.stack(displacement_samples).astype(np.float32),
        "fixed_mask": fixed_mask(mesh).astype(np.float32),
        "condition": np.stack(condition_samples).astype(np.float32),
        "compliance": np.asarray(
            compliance_samples,
            dtype=np.float32,
        ),
        "relative_residual": np.asarray(
            residual_samples,
            dtype=np.float64,
        ),
        "condition_id": np.asarray(
            condition_id_samples,
            dtype=np.int16,
        ),
        "split": np.asarray(split_samples, dtype=np.uint8),
        "iteration": np.asarray(
            iteration_samples,
            dtype=np.int16,
        ),
        "progress": np.asarray(
            progress_samples,
            dtype=np.float32,
        ),
        "snapshot_rank": np.asarray(
            snapshot_rank_samples,
            dtype=np.uint8,
        ),
    }

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **arrays)

    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    with conditions_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "condition_id",
                "split",
                "target_volume",
                "load_y_fraction",
                "load_angle_deg",
                "iterations",
                "converged",
                "final_change",
                "terminal_compliance",
                "runtime_seconds",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(condition_rows)

    write_density_montage(
        montage_densities,
        mesh,
        figure_path,
    )

    condition_counts = {
        split: sum(
            1 for condition in conditions
            if condition.split == split
        )
        for split in ("train", "val", "test")
    }

    sample_counts = {
        split: int(
            np.count_nonzero(
                arrays["split"] == SPLIT_CODE[split]
            )
        )
        for split in ("train", "val", "test")
    }

    terminal_compliances = np.array(
        [
            float(row["terminal_compliance"])
            for row in condition_rows
        ],
        dtype=float,
    )
    iterations = np.array(
        [int(row["iterations"]) for row in condition_rows],
        dtype=int,
    )

    manifest: dict[str, object] = {
        "stage": "V04",
        "dataset_schema": 1,
        "dataset_file": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_bytes": dataset_path.stat().st_size,
        "mesh": {
            "nelx": mesh.nelx,
            "nely": mesh.nely,
            "nodes_x": mesh.nelx + 1,
            "nodes_y": mesh.nely + 1,
        },
        "physics": {
            "young": 1.0,
            "poisson": 0.3,
            "plane_stress": True,
            "load_magnitude": 1.0,
        },
        "simp": {
            "penal": config.penal,
            "rmin_elements": config.rmin,
            "emin_ratio": config.emin_ratio,
            "move": config.move,
            "tolerance": config.tolerance,
            "max_iterations": config.max_iterations,
        },
        "condition_grid": {
            "target_volume": [0.30, 0.40, 0.50],
            "load_y_fraction": [0.25, 0.50, 0.75],
            "load_angle_deg": [-105.0, -90.0, -75.0],
            "number_of_conditions": len(conditions),
        },
        "split": {
            "seed": SPLIT_SEED,
            "condition_counts": condition_counts,
            "sample_counts": sample_counts,
            "split_is_by_condition": True,
        },
        "snapshots": {
            "fractions": list(SNAPSHOT_FRACTIONS),
            "per_condition": len(SNAPSHOT_FRACTIONS),
            "total_samples": int(arrays["condition_id"].size),
        },
        "arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in arrays.items()
        },
        "quality": {
            "all_conditions_converged": all(
                bool(row["converged"])
                for row in condition_rows
            ),
            "max_relative_residual": float(
                np.max(arrays["relative_residual"])
            ),
            "iteration_min": int(np.min(iterations)),
            "iteration_max": int(np.max(iterations)),
            "terminal_compliance_min": float(
                np.min(terminal_compliances)
            ),
            "terminal_compliance_max": float(
                np.max(terminal_compliances)
            ),
        },
        "runtime_seconds": time.perf_counter() - start_all,
        "notes": (
            "The NPZ is generated locally and ignored by Git. "
            "Tracked manifest and conditions.csv are sufficient to audit "
            "the split, schema and exact dataset SHA256."
        ),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    """Загрузить NPZ без pickle и вернуть обычный словарь."""

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
