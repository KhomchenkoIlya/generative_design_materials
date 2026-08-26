"""V08: exact FEM screening and fixed-budget SIMP refinement of DDPM candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import zlib
from collections import deque

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import (  # noqa: E402
    StructuredQuadMesh,
    assemble_stiffness,
    cantilever_condition,
    solve_elasticity,
)
from project1_topopt.topopt import (  # noqa: E402
    SIMPConfig,
    analyze_design,
    build_density_filter,
    oc_update,
)


SCREEN_COUNT_EXPECTED = 80
TOP_K = 4
REFINEMENT_STEPS = 30
CONNECTIVITY_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upsample_density_2x(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("field должен иметь shape (H,W).")
    return np.repeat(
        np.repeat(field, 2, axis=0),
        2,
        axis=1,
    )


def field_to_design(
    field: np.ndarray,
    mesh: StructuredQuadMesh,
) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    expected = (mesh.nely, mesh.nelx)
    if field.shape != expected:
        raise ValueError(
            f"ожидалась density shape {expected}, получено {field.shape}."
        )
    return field.T.reshape(-1).copy()


def design_to_field(
    design: np.ndarray,
    mesh: StructuredQuadMesh,
) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    if design.shape != (mesh.nelems,):
        raise ValueError("design должен иметь длину mesh.nelems.")
    return design.reshape(mesh.nelx, mesh.nely).T.copy()


def binarity_score(field: np.ndarray) -> float:
    rho = np.asarray(field, dtype=np.float64)
    return float(
        np.mean(
            4.0 * rho * (1.0 - rho)
        )
    )


def support_to_load_path(
    field: np.ndarray,
    load_y_fraction: float,
    *,
    threshold: float = CONNECTIVITY_THRESHOLD,
) -> bool:
    """4-neighbour path through rho >= threshold from support to loaded edge."""

    rho = np.asarray(field, dtype=np.float64)
    if rho.ndim != 2:
        raise ValueError("field должен быть двумерным.")

    nely, nelx = rho.shape
    solid = rho >= threshold

    starts = [
        (iy, 0)
        for iy in range(nely)
        if solid[iy, 0]
    ]
    if not starts:
        return False

    load_node_y = int(round(load_y_fraction * nely))
    target_rows = {
        min(nely - 1, max(0, load_node_y - 1)),
        min(nely - 1, max(0, load_node_y)),
    }
    targets = {
        (iy, nelx - 1)
        for iy in target_rows
        if solid[iy, nelx - 1]
    }
    if not targets:
        return False

    queue: deque[tuple[int, int]] = deque(starts)
    visited = set(starts)

    while queue:
        iy, ix = queue.popleft()
        if (iy, ix) in targets:
            return True

        for dy, dx in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            jy = iy + dy
            jx = ix + dx
            if not (0 <= jy < nely and 0 <= jx < nelx):
                continue
            if not solid[jy, jx]:
                continue
            point = (jy, jx)
            if point in visited:
                continue
            visited.add(point)
            queue.append(point)

    return False


def summary(values: np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("summary требует непустой массив.")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def direct_fem(
    mesh: StructuredQuadMesh,
    problem,
    field: np.ndarray,
    config: SIMPConfig,
) -> dict[str, float]:
    rho = field_to_design(field, mesh)
    if np.any(rho < 0.0) or np.any(rho > 1.0):
        raise ValueError("physical density должна лежать в [0,1].")

    factors = (
        config.emin_ratio
        + rho**config.penal
        * (1.0 - config.emin_ratio)
    )
    stiffness = assemble_stiffness(
        mesh,
        young=1.0,
        poisson=0.3,
        thickness=1.0,
        element_factors=factors,
    )
    fem = solve_elasticity(
        stiffness,
        problem,
    )

    return {
        "compliance": float(fem.compliance),
        "relative_residual": float(fem.relative_residual),
        "energy_identity_error": float(fem.energy_identity_error),
        "force_balance_error": float(fem.force_balance_error),
        "volume_fraction": float(np.mean(rho)),
    }


def fixed_budget_refinement(
    mesh: StructuredQuadMesh,
    problem,
    initial_field: np.ndarray,
    config: SIMPConfig,
    *,
    steps: int = REFINEMENT_STEPS,
) -> dict[str, object]:
    """Exactly `steps` OC updates from the supplied warm start."""

    if steps <= 0:
        raise ValueError("steps должен быть положительным.")

    density_filter = build_density_filter(
        mesh,
        config.rmin,
    )
    design = np.clip(
        field_to_design(
            initial_field,
            mesh,
        ),
        0.0,
        1.0,
    )

    analysis = analyze_design(
        mesh,
        problem,
        design,
        density_filter,
        config,
    )
    start_compliance = float(
        analysis.compliance
    )
    start_volume = float(
        analysis.volume_fraction
    )

    history: list[dict[str, float]] = []

    for iteration in range(1, steps + 1):
        updated = oc_update(
            design,
            analysis.gradient,
            analysis.volume_gradient,
            density_filter,
            problem.volume_fraction,
            config.move,
        )
        change = float(
            np.max(
                np.abs(updated - design)
            )
        )
        design = updated

        analysis = analyze_design(
            mesh,
            problem,
            design,
            density_filter,
            config,
        )

        history.append(
            {
                "iteration": float(iteration),
                "compliance": float(
                    analysis.compliance
                ),
                "volume_fraction": float(
                    analysis.volume_fraction
                ),
                "max_change": change,
            }
        )

    return {
        "design": design,
        "physical_density": analysis.physical_density.copy(),
        "field": design_to_field(
            analysis.physical_density,
            mesh,
        ),
        "start_filtered_compliance": start_compliance,
        "start_filtered_volume": start_volume,
        "compliance": float(analysis.compliance),
        "volume_fraction": float(analysis.volume_fraction),
        "relative_residual": float(analysis.relative_residual),
        "energy_identity_error": float(analysis.energy_identity_error),
        "force_balance_error": float(analysis.force_balance_error),
        "final_change": float(
            history[-1]["max_change"]
        ),
        "history": tuple(history),
    }


def _png_chunk(
    kind: bytes,
    data: bytes,
) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(
            ">I",
            zlib.crc32(kind + data) & 0xFFFFFFFF,
        )
    )


def write_comparison_montage(
    blocks: np.ndarray,
    path: Path,
    *,
    scale: int = 3,
    gap: int = 3,
) -> None:
    """blocks shape = (conditions, 4, 32, 96).

    Columns: best screened, best refined, uniform-30, converged reference.
    """

    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.ndim != 4 or blocks.shape[1] != 4:
        raise ValueError(
            "blocks должен иметь shape (conditions,4,H,W)."
        )

    rows, columns, height, width = blocks.shape

    tile_h = height * scale
    tile_w = width * scale
    canvas_h = (
        rows * tile_h
        + (rows - 1) * gap
    )
    canvas_w = (
        columns * tile_w
        + (columns - 1) * gap
    )

    canvas = np.full(
        (canvas_h, canvas_w),
        255,
        dtype=np.uint8,
    )

    for row in range(rows):
        for column in range(columns):
            field = np.clip(
                blocks[row, column],
                0.0,
                1.0,
            )
            gray = np.rint(
                255.0 * (1.0 - np.flipud(field))
            ).astype(np.uint8)
            tile = np.repeat(
                np.repeat(
                    gray,
                    scale,
                    axis=0,
                ),
                scale,
                axis=1,
            )
            y0 = row * (tile_h + gap)
            x0 = column * (tile_w + gap)
            canvas[
                y0 : y0 + tile_h,
                x0 : x0 + tile_w,
            ] = tile

    raw = b"".join(
        b"\x00" + canvas[row].tobytes()
        for row in range(canvas.shape[0])
    )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                canvas.shape[1],
                canvas.shape[0],
                8,
                0,
                0,
                0,
                0,
            ),
        )
        + _png_chunk(
            b"IDAT",
            zlib.compress(raw, level=9),
        )
        + _png_chunk(b"IEND", b"")
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_bytes(png)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def angle_from_condition(
    condition: np.ndarray,
) -> float:
    return math.degrees(
        math.atan2(
            float(condition[3]),
            float(condition[2]),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--screening-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--refinement-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--refined-npz",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--figure",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    start_all = time.perf_counter()

    with np.load(
        args.candidates,
        allow_pickle=False,
    ) as archive:
        projected = archive[
            "projected_density"
        ].astype(np.float64)
        generated_condition = archive[
            "condition"
        ].astype(np.float64)
        generated_condition_id = archive[
            "condition_id"
        ].astype(np.int16)

    with np.load(
        args.dataset,
        allow_pickle=False,
    ) as archive:
        dataset_rho = archive[
            "rho_element"
        ].astype(np.float64)
        dataset_condition = archive[
            "condition"
        ].astype(np.float64)
        dataset_condition_id = archive[
            "condition_id"
        ].astype(np.int16)
        dataset_split = archive[
            "split"
        ].astype(np.uint8)
        dataset_snapshot_rank = archive[
            "snapshot_rank"
        ].astype(np.uint8)
        dataset_compliance = archive[
            "compliance"
        ].astype(np.float64)

    if projected.shape != (
        5,
        16,
        16,
        48,
    ):
        raise RuntimeError(
            "ожидались V07 candidates shape (5,16,16,48), "
            f"получено {projected.shape}."
        )
    if generated_condition.shape != (5, 4):
        raise RuntimeError(
            "ожидались 5 engineering conditions."
        )

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
        max_iterations=REFINEMENT_STEPS,
    )

    screening_rows: list[
        dict[str, object]
    ] = []
    screening_records: dict[
        int,
        list[dict[str, object]],
    ] = {}

    max_screen_residual = 0.0

    print(
        "V08 screening: exact FEM for "
        f"{projected.shape[0] * projected.shape[1]} candidates",
        flush=True,
    )

    for condition_ordinal in range(
        projected.shape[0]
    ):
        cid = int(
            generated_condition_id[
                condition_ordinal
            ]
        )
        c = generated_condition[
            condition_ordinal
        ]
        target_volume = float(c[0])
        load_y = float(c[1])
        angle = angle_from_condition(c)

        problem = cantilever_condition(
            mesh,
            volume_fraction=target_volume,
            load_y_fraction=load_y,
            load_angle_deg=angle,
            load_magnitude=1.0,
        )

        records: list[
            dict[str, object]
        ] = []

        for candidate_index in range(
            projected.shape[1]
        ):
            field = upsample_density_2x(
                projected[
                    condition_ordinal,
                    candidate_index,
                ]
            )
            fem = direct_fem(
                mesh,
                problem,
                field,
                config,
            )
            max_screen_residual = max(
                max_screen_residual,
                fem["relative_residual"],
            )

            connected = support_to_load_path(
                field,
                load_y,
            )
            binarity = binarity_score(
                field
            )

            record = {
                "condition_id": cid,
                "candidate_index": candidate_index,
                "target_volume": target_volume,
                "load_y_fraction": load_y,
                "load_angle_deg": angle,
                "compliance": fem["compliance"],
                "volume_fraction": fem["volume_fraction"],
                "volume_error": abs(
                    fem["volume_fraction"]
                    - target_volume
                ),
                "relative_residual": fem["relative_residual"],
                "binarity": binarity,
                "connected_at_0_5": int(
                    connected
                ),
                "field": field,
            }
            records.append(record)

            screening_rows.append(
                {
                    key: value
                    for key, value in record.items()
                    if key != "field"
                }
            )

        records.sort(
            key=lambda item: float(
                item["compliance"]
            )
        )
        screening_records[cid] = records

        print(
            f"  condition={cid}: "
            f"best J={float(records[0]['compliance']):.6f}, "
            f"connected={sum(int(r['connected_at_0_5']) for r in records)}/16",
            flush=True,
        )

    if len(screening_rows) != SCREEN_COUNT_EXPECTED:
        raise RuntimeError(
            f"ожидалось {SCREEN_COUNT_EXPECTED} screening rows."
        )

    refinement_rows: list[
        dict[str, object]
    ] = []
    condition_metrics: list[
        dict[str, object]
    ] = []

    selected_initial_all: list[
        np.ndarray
    ] = []
    selected_refined_all: list[
        np.ndarray
    ] = []
    selected_indices_all: list[
        np.ndarray
    ] = []
    uniform30_all: list[
        np.ndarray
    ] = []
    terminal_all: list[
        np.ndarray
    ] = []
    montage_rows: list[
        np.ndarray
    ] = []

    initial_to_refined_ratios: list[float] = []
    refinement_field_changes: list[float] = []
    best_over_uniform_ratios: list[float] = []
    best_over_terminal_ratios: list[float] = []
    uniform_over_terminal_ratios: list[float] = []
    best_screened_over_terminal_ratios: list[float] = []

    initial_selected_binarity: list[float] = []
    refined_selected_binarity: list[float] = []
    terminal_binarity: list[float] = []

    selected_connected_before = 0
    selected_connected_after = 0
    diffusion_better_uniform_count = 0

    max_refine_residual = 0.0
    max_volume_error = 0.0

    print(
        "V08 refinement: top-4 per condition, "
        "30 fixed OC updates; plus uniform 30-step baseline",
        flush=True,
    )

    for condition_ordinal in range(
        generated_condition.shape[0]
    ):
        cid = int(
            generated_condition_id[
                condition_ordinal
            ]
        )
        c = generated_condition[
            condition_ordinal
        ]
        target_volume = float(c[0])
        load_y = float(c[1])
        angle = angle_from_condition(c)

        problem = cantilever_condition(
            mesh,
            volume_fraction=target_volume,
            load_y_fraction=load_y,
            load_angle_deg=angle,
            load_magnitude=1.0,
        )

        terminal_mask = (
            (dataset_split == 2)
            & (dataset_condition_id == cid)
            & (dataset_snapshot_rank == 5)
        )
        terminal_indices = np.flatnonzero(
            terminal_mask
        )
        if terminal_indices.size != 1:
            raise RuntimeError(
                f"condition {cid}: нужен ровно один terminal test sample."
            )
        terminal_index = int(
            terminal_indices[0]
        )

        if not np.allclose(
            dataset_condition[terminal_index],
            c,
            atol=2.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"condition {cid}: candidate condition не совпадает с V04."
            )

        terminal_field = dataset_rho[
            terminal_index
        ]
        terminal_j = float(
            dataset_compliance[
                terminal_index
            ]
        )

        terminal_check = direct_fem(
            mesh,
            problem,
            terminal_field,
            config,
        )
        if abs(
            terminal_check["compliance"]
            - terminal_j
        ) / max(1.0, abs(terminal_j)) > 1.0e-5:
            raise RuntimeError(
                f"condition {cid}: terminal compliance не воспроизводится."
            )

        uniform_field = np.full(
            (mesh.nely, mesh.nelx),
            target_volume,
            dtype=np.float64,
        )
        uniform30 = fixed_budget_refinement(
            mesh,
            problem,
            uniform_field,
            config,
            steps=REFINEMENT_STEPS,
        )
        uniform_j = float(
            uniform30["compliance"]
        )
        uniform_field_final = np.asarray(
            uniform30["field"],
            dtype=np.float64,
        )

        max_refine_residual = max(
            max_refine_residual,
            float(
                uniform30[
                    "relative_residual"
                ]
            ),
        )
        max_volume_error = max(
            max_volume_error,
            abs(
                float(
                    uniform30[
                        "volume_fraction"
                    ]
                )
                - target_volume
            ),
        )

        selected = screening_records[
            cid
        ][:TOP_K]
        selected_indices = np.asarray(
            [
                int(item["candidate_index"])
                for item in selected
            ],
            dtype=np.int16,
        )

        selected_initial: list[
            np.ndarray
        ] = []
        selected_refined: list[
            np.ndarray
        ] = []

        refined_records: list[
            dict[str, object]
        ] = []

        for rank, screened in enumerate(
            selected,
            start=1,
        ):
            initial_field = np.asarray(
                screened["field"],
                dtype=np.float64,
            )

            refined = fixed_budget_refinement(
                mesh,
                problem,
                initial_field,
                config,
                steps=REFINEMENT_STEPS,
            )
            refined_field = np.asarray(
                refined["field"],
                dtype=np.float64,
            )
            refined_j = float(
                refined["compliance"]
            )

            max_refine_residual = max(
                max_refine_residual,
                float(
                    refined[
                        "relative_residual"
                    ]
                ),
            )
            max_volume_error = max(
                max_volume_error,
                abs(
                    float(
                        refined[
                            "volume_fraction"
                        ]
                    )
                    - target_volume
                ),
            )

            connected_before = bool(
                screened[
                    "connected_at_0_5"
                ]
            )
            connected_after = support_to_load_path(
                refined_field,
                load_y,
            )

            selected_connected_before += int(
                connected_before
            )
            selected_connected_after += int(
                connected_after
            )

            initial_b = float(
                screened["binarity"]
            )
            refined_b = binarity_score(
                refined_field
            )
            initial_selected_binarity.append(
                initial_b
            )
            refined_selected_binarity.append(
                refined_b
            )

            initial_j = float(
                screened["compliance"]
            )
            compliance_ratio = (
                refined_j / initial_j
            )
            initial_to_refined_ratios.append(
                compliance_ratio
            )

            field_change = float(
                np.sqrt(
                    np.mean(
                        (
                            refined_field
                            - initial_field
                        )
                        ** 2
                    )
                )
            )
            refinement_field_changes.append(
                field_change
            )

            row = {
                "condition_id": cid,
                "screen_rank": rank,
                "candidate_index": int(
                    screened[
                        "candidate_index"
                    ]
                ),
                "screen_compliance": initial_j,
                "filtered_start_compliance": float(
                    refined[
                        "start_filtered_compliance"
                    ]
                ),
                "refined_compliance": refined_j,
                "uniform30_compliance": uniform_j,
                "terminal_compliance": terminal_j,
                "refined_over_screen": compliance_ratio,
                "refined_over_uniform30": refined_j / uniform_j,
                "refined_over_terminal": refined_j / terminal_j,
                "initial_binarity": initial_b,
                "refined_binarity": refined_b,
                "rms_density_change": field_change,
                "connected_before": int(
                    connected_before
                ),
                "connected_after": int(
                    connected_after
                ),
                "final_volume_fraction": float(
                    refined[
                        "volume_fraction"
                    ]
                ),
                "final_relative_residual": float(
                    refined[
                        "relative_residual"
                    ]
                ),
                "final_change": float(
                    refined[
                        "final_change"
                    ]
                ),
            }
            refinement_rows.append(
                row
            )
            refined_records.append(
                {
                    **row,
                    "field": refined_field,
                }
            )
            selected_initial.append(
                initial_field
            )
            selected_refined.append(
                refined_field
            )

        refined_records.sort(
            key=lambda item: float(
                item[
                    "refined_compliance"
                ]
            )
        )
        best_refined = refined_records[0]
        best_refined_j = float(
            best_refined[
                "refined_compliance"
            ]
        )
        best_screened_j = float(
            selected[0][
                "compliance"
            ]
        )

        best_over_uniform = (
            best_refined_j / uniform_j
        )
        best_over_terminal = (
            best_refined_j / terminal_j
        )
        uniform_over_terminal = (
            uniform_j / terminal_j
        )
        best_screened_over_terminal = (
            best_screened_j / terminal_j
        )

        best_over_uniform_ratios.append(
            best_over_uniform
        )
        best_over_terminal_ratios.append(
            best_over_terminal
        )
        uniform_over_terminal_ratios.append(
            uniform_over_terminal
        )
        best_screened_over_terminal_ratios.append(
            best_screened_over_terminal
        )

        if best_refined_j < uniform_j:
            diffusion_better_uniform_count += 1

        terminal_b = binarity_score(
            terminal_field
        )
        terminal_binarity.append(
            terminal_b
        )

        condition_metrics.append(
            {
                "condition_id": cid,
                "target_volume": target_volume,
                "load_y_fraction": load_y,
                "load_angle_deg": angle,
                "best_screen_candidate_index": int(
                    selected[0][
                        "candidate_index"
                    ]
                ),
                "best_screen_compliance": best_screened_j,
                "best_refined_candidate_index": int(
                    best_refined[
                        "candidate_index"
                    ]
                ),
                "best_refined_compliance": best_refined_j,
                "uniform30_compliance": uniform_j,
                "terminal_compliance": terminal_j,
                "best_refined_over_uniform30": best_over_uniform,
                "best_refined_over_terminal": best_over_terminal,
                "uniform30_over_terminal": uniform_over_terminal,
                "best_screened_over_terminal": best_screened_over_terminal,
                "terminal_binarity": terminal_b,
            }
        )

        selected_initial_array = np.stack(
            selected_initial,
            axis=0,
        )
        selected_refined_array = np.stack(
            selected_refined,
            axis=0,
        )

        selected_initial_all.append(
            selected_initial_array
        )
        selected_refined_all.append(
            selected_refined_array
        )
        selected_indices_all.append(
            selected_indices
        )
        uniform30_all.append(
            uniform_field_final
        )
        terminal_all.append(
            terminal_field
        )

        best_refined_field = np.asarray(
            best_refined["field"],
            dtype=np.float64,
        )

        montage_rows.append(
            np.stack(
                (
                    np.asarray(
                        selected[0]["field"],
                        dtype=np.float64,
                    ),
                    best_refined_field,
                    uniform_field_final,
                    terminal_field,
                ),
                axis=0,
            )
        )

        print(
            f"  condition={cid}: "
            f"Jgen={best_screened_j:.6f}, "
            f"Jref={best_refined_j:.6f}, "
            f"Juniform30={uniform_j:.6f}, "
            f"Jterminal={terminal_j:.6f}",
            flush=True,
        )

    selected_initial_array = np.stack(
        selected_initial_all,
        axis=0,
    )
    selected_refined_array = np.stack(
        selected_refined_all,
        axis=0,
    )
    selected_indices_array = np.stack(
        selected_indices_all,
        axis=0,
    )
    uniform30_array = np.stack(
        uniform30_all,
        axis=0,
    )
    terminal_array = np.stack(
        terminal_all,
        axis=0,
    )
    montage_array = np.stack(
        montage_rows,
        axis=0,
    )

    args.refined_npz.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        args.refined_npz,
        condition_id=generated_condition_id,
        condition=generated_condition.astype(
            np.float32
        ),
        selected_candidate_index=selected_indices_array,
        selected_initial_density=selected_initial_array.astype(
            np.float32
        ),
        selected_refined_density=selected_refined_array.astype(
            np.float32
        ),
        uniform30_density=uniform30_array.astype(
            np.float32
        ),
        terminal_reference_density=terminal_array.astype(
            np.float32
        ),
    )

    write_comparison_montage(
        montage_array,
        args.figure,
    )

    write_csv(
        args.screening_csv,
        screening_rows,
        (
            "condition_id",
            "candidate_index",
            "target_volume",
            "load_y_fraction",
            "load_angle_deg",
            "compliance",
            "volume_fraction",
            "volume_error",
            "relative_residual",
            "binarity",
            "connected_at_0_5",
        ),
    )

    write_csv(
        args.refinement_csv,
        refinement_rows,
        (
            "condition_id",
            "screen_rank",
            "candidate_index",
            "screen_compliance",
            "filtered_start_compliance",
            "refined_compliance",
            "uniform30_compliance",
            "terminal_compliance",
            "refined_over_screen",
            "refined_over_uniform30",
            "refined_over_terminal",
            "initial_binarity",
            "refined_binarity",
            "rms_density_change",
            "connected_before",
            "connected_after",
            "final_volume_fraction",
            "final_relative_residual",
            "final_change",
        ),
    )

    screening_compliance = np.asarray(
        [
            float(row["compliance"])
            for row in screening_rows
        ],
        dtype=np.float64,
    )
    screening_binarity = np.asarray(
        [
            float(row["binarity"])
            for row in screening_rows
        ],
        dtype=np.float64,
    )
    screening_connected = sum(
        int(row["connected_at_0_5"])
        for row in screening_rows
    )

    runtime_seconds = (
        time.perf_counter() - start_all
    )

    result = {
        "stage": "V08",
        "source_candidate_npz_sha256": sha256_file(
            args.candidates
        ),
        "refined_npz_sha256": sha256_file(
            args.refined_npz
        ),
        "refined_npz_bytes": args.refined_npz.stat().st_size,
        "protocol": {
            "mesh_elements": [96, 32],
            "screen_candidates": len(
                screening_rows
            ),
            "top_k_per_condition": TOP_K,
            "refined_candidates": len(
                refinement_rows
            ),
            "simp_steps": REFINEMENT_STEPS,
            "uniform_baseline_steps": REFINEMENT_STEPS,
            "penal": config.penal,
            "rmin": config.rmin,
            "emin_ratio": config.emin_ratio,
            "move": config.move,
            "connectivity_threshold": CONNECTIVITY_THRESHOLD,
        },
        "screening": {
            "compliance": summary(
                screening_compliance
            ),
            "binarity": summary(
                screening_binarity
            ),
            "connected_at_0_5_count": screening_connected,
            "connected_at_0_5_fraction": (
                screening_connected
                / len(screening_rows)
            ),
            "max_volume_error": float(
                max(
                    float(row["volume_error"])
                    for row in screening_rows
                )
            ),
            "max_relative_residual": max_screen_residual,
        },
        "refinement": {
            "initial_to_refined_compliance_ratio": summary(
                initial_to_refined_ratios
            ),
            "best_refined_over_uniform30": summary(
                best_over_uniform_ratios
            ),
            "best_refined_over_terminal": summary(
                best_over_terminal_ratios
            ),
            "uniform30_over_terminal": summary(
                uniform_over_terminal_ratios
            ),
            "best_screened_over_terminal": summary(
                best_screened_over_terminal_ratios
            ),
            "diffusion_better_uniform30_count": diffusion_better_uniform_count,
            "condition_count": len(
                condition_metrics
            ),
            "selected_connected_before_count": selected_connected_before,
            "selected_connected_after_count": selected_connected_after,
            "selected_count": len(
                refinement_rows
            ),
            "initial_selected_binarity": summary(
                initial_selected_binarity
            ),
            "refined_selected_binarity": summary(
                refined_selected_binarity
            ),
            "terminal_binarity": summary(
                terminal_binarity
            ),
            "rms_density_change": summary(
                refinement_field_changes
            ),
            "max_final_volume_error": max_volume_error,
            "max_relative_residual": max_refine_residual,
        },
        "condition_metrics": condition_metrics,
        "runtime_seconds": runtime_seconds,
        "interpretation_guardrails": {
            "all_80_ranked_by_exact_fem": True,
            "only_top4_refined_per_condition": True,
            "same_30_step_budget_for_diffusion_and_uniform_baseline": True,
            "terminal_reference_is_fully_converged_v04_simp": True,
            "connectivity_is_descriptive_threshold_metric": True,
        },
    }

    args.result.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.result.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    warm = result["refinement"][
        "best_refined_over_uniform30"
    ]
    gap = result["refinement"][
        "best_refined_over_terminal"
    ]

    print("V08 exact FEM + SIMP refinement finished")
    print(
        "diffusion better than uniform-30: "
        f"{diffusion_better_uniform_count}/"
        f"{len(condition_metrics)}"
    )
    print(
        "best refined / uniform-30: "
        f"median={warm['median']:.4f}, "
        f"p90={warm['p90']:.4f}"
    )
    print(
        "best refined / terminal: "
        f"median={gap['median']:.4f}, "
        f"p90={gap['p90']:.4f}"
    )
    print(
        "binarity selected initial/refined/terminal: "
        f"{np.median(initial_selected_binarity):.4f}/"
        f"{np.median(refined_selected_binarity):.4f}/"
        f"{np.median(terminal_binarity):.4f}"
    )
    print(
        "connectivity selected before/after: "
        f"{selected_connected_before}/"
        f"{len(refinement_rows)} -> "
        f"{selected_connected_after}/"
        f"{len(refinement_rows)}"
    )
    print(
        f"runtime={runtime_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
