"""V05: обучение компактного FNO и проверка против точного FEM."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
import time
import zlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import (  # noqa: E402
    StructuredQuadMesh,
    assemble_stiffness,
    cantilever_condition,
    solve_elasticity,
)
from project1_topopt.topopt import resample_factor_two  # noqa: E402
from project2_operator.dataset import (  # noqa: E402
    element_to_node_density,
    force_to_grid,
)
from project2_operator.fno import (  # noqa: E402
    FNO2d,
    build_raw_inputs,
    channel_statistics,
    count_parameters,
    output_rms,
)


SEED = 20260814
TRAIN_SPLIT = 0
VAL_SPLIT = 1
TEST_SPLIT = 2


def set_reproducible_cpu(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True)


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    diff = torch.linalg.vector_norm(
        (prediction - target).flatten(1),
        dim=1,
    )
    base = torch.linalg.vector_norm(
        target.flatten(1),
        dim=1,
    )
    return torch.mean(diff / torch.clamp(base, min=1.0e-12))


def array_relative_l2(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    diff = np.linalg.norm(
        (prediction - target).reshape(prediction.shape[0], -1),
        axis=1,
    )
    base = np.linalg.norm(
        target.reshape(target.shape[0], -1),
        axis=1,
    )
    return diff / np.maximum(base, 1.0e-12)


def inverse_grid(
    grid: np.ndarray,
) -> np.ndarray:
    """(2,H,W) -> FEM vector with node-major interleaved components."""

    return np.transpose(grid, (2, 1, 0)).reshape(-1)


def simp_factors(
    rho: np.ndarray,
    *,
    penal: float = 3.0,
    emin_ratio: float = 1.0e-9,
) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    return emin_ratio + rho**penal * (1.0 - emin_ratio)


def fixed_dofs_for_cantilever(
    mesh: StructuredQuadMesh,
) -> np.ndarray:
    left_nodes = np.arange(mesh.nely + 1, dtype=int)
    fixed = np.empty(2 * left_nodes.size, dtype=int)
    fixed[0::2] = 2 * left_nodes
    fixed[1::2] = 2 * left_nodes + 1
    return fixed


def physical_metrics(
    rho_element: np.ndarray,
    force_grid: np.ndarray,
    predicted_grid: np.ndarray,
    true_compliance: float,
    mesh: StructuredQuadMesh,
) -> tuple[float, float]:
    """Relative compliance error и free-DOF equilibrium residual."""

    rho_vector = np.asarray(rho_element, dtype=float).T.reshape(-1)
    stiffness = assemble_stiffness(
        mesh,
        young=1.0,
        poisson=0.3,
        thickness=1.0,
        element_factors=simp_factors(rho_vector),
    )

    predicted = inverse_grid(predicted_grid)
    force = inverse_grid(force_grid)

    fixed = fixed_dofs_for_cantilever(mesh)
    all_dofs = np.arange(2 * mesh.nnodes)
    free = np.setdiff1d(
        all_dofs,
        fixed,
        assume_unique=True,
    )

    residual = stiffness[free, :] @ predicted - force[free]
    denominator = max(np.linalg.norm(force[free]), 1.0e-12)
    relative_residual = float(
        np.linalg.norm(residual) / denominator
    )

    predicted_compliance = float(force @ predicted)
    compliance_error = abs(
        predicted_compliance - true_compliance
    ) / max(abs(true_compliance), 1.0e-12)

    return float(compliance_error), relative_residual


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_comparison_png(
    truth: np.ndarray,
    prediction: np.ndarray,
    path: Path,
    *,
    scale: int = 4,
    gap: int = 5,
) -> None:
    """Три панели: |U| exact, |U| FNO, |error|."""

    truth_mag = np.sqrt(np.sum(np.asarray(truth) ** 2, axis=0))
    pred_mag = np.sqrt(np.sum(np.asarray(prediction) ** 2, axis=0))
    err_mag = np.sqrt(np.sum((np.asarray(prediction) - truth) ** 2, axis=0))

    common_max = max(
        float(np.max(truth_mag)),
        float(np.max(pred_mag)),
        1.0e-12,
    )
    error_max = max(float(np.max(err_mag)), 1.0e-12)

    panels = [
        truth_mag / common_max,
        pred_mag / common_max,
        err_mag / error_max,
    ]

    images: list[np.ndarray] = []
    for panel in panels:
        gray = np.rint(
            255.0 * (1.0 - np.clip(panel, 0.0, 1.0))
        ).astype(np.uint8)
        gray = np.flipud(gray)
        image = np.repeat(
            np.repeat(gray, scale, axis=0),
            scale,
            axis=1,
        )
        images.append(image)

    height = images[0].shape[0]
    separator = np.full((height, gap), 255, dtype=np.uint8)
    canvas = np.concatenate(
        [
            images[0],
            separator,
            images[1],
            separator,
            images[2],
        ],
        axis=1,
    )

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


def evaluate_model(
    model: nn.Module,
    inputs: np.ndarray,
    target: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_scale: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []

    normalized = (
        inputs[indices] - input_mean
    ) / input_std

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(start + batch_size, len(indices))
            x = torch.from_numpy(
                normalized[start:stop]
            ).float()
            pred_scaled = model(x)
            pred = (
                pred_scaled.cpu().numpy()
                * output_scale
            )
            predictions.append(pred)

    return np.concatenate(predictions, axis=0)


def benchmark_fno(
    model: nn.Module,
    normalized_input: np.ndarray,
    output_scale: np.ndarray,
    repeats: int = 40,
) -> float:
    x = torch.from_numpy(normalized_input[None, ...]).float()

    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)

        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        elapsed = time.perf_counter() - start

    _ = output_scale
    return 1000.0 * elapsed / repeats


def benchmark_fem(
    rho_element: np.ndarray,
    force_grid: np.ndarray,
    mesh: StructuredQuadMesh,
    repeats: int = 3,
) -> float:
    rho_vector = np.asarray(rho_element, dtype=float).T.reshape(-1)
    factors = simp_factors(rho_vector)
    force = inverse_grid(force_grid)

    # Нужен только ProblemCondition для существующего solve_elasticity.
    # Параметры нагрузки восстанавливать не требуется: force уже точный.
    problem = cantilever_condition(
        mesh,
        volume_fraction=float(np.mean(rho_element)),
        load_y_fraction=0.5,
        load_angle_deg=-90.0,
        load_magnitude=1.0,
    )

    # Заменяем только force неизменяемого dataclass через его тип.
    fields = problem.__dataclass_fields__
    kwargs = {
        name: getattr(problem, name)
        for name in fields
    }
    kwargs["force"] = force
    problem = type(problem)(**kwargs)

    start = time.perf_counter()
    for _ in range(repeats):
        stiffness = assemble_stiffness(
            mesh,
            young=1.0,
            poisson=0.3,
            thickness=1.0,
            element_factors=factors,
        )
        _ = solve_elasticity(stiffness, problem)
    elapsed = time.perf_counter() - start

    return 1000.0 * elapsed / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    set_reproducible_cpu(SEED)

    with np.load(args.dataset, allow_pickle=False) as archive:
        rho_element = archive["rho_element"].astype(np.float32)
        rho_node = archive["rho_node"].astype(np.float32)
        force = archive["force"].astype(np.float32)
        displacement = archive["displacement"].astype(np.float32)
        condition = archive["condition"].astype(np.float32)
        compliance = archive["compliance"].astype(np.float64)
        split = archive["split"].astype(np.uint8)
        condition_id = archive["condition_id"].astype(np.int16)
        snapshot_rank = archive["snapshot_rank"].astype(np.uint8)

    raw_inputs = build_raw_inputs(
        rho_node,
        force,
        condition,
    )

    train_mask = split == TRAIN_SPLIT
    val_mask = split == VAL_SPLIT
    test_mask = split == TEST_SPLIT

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    input_mean, input_std = channel_statistics(
        raw_inputs,
        train_mask,
    )
    scale = output_rms(
        displacement,
        train_mask,
    )

    normalized_inputs = (
        raw_inputs - input_mean
    ) / input_std
    scaled_targets = displacement / scale

    train_x = torch.from_numpy(
        normalized_inputs[train_idx]
    ).float()
    train_y = torch.from_numpy(
        scaled_targets[train_idx]
    ).float()
    val_x = torch.from_numpy(
        normalized_inputs[val_idx]
    ).float()
    val_y = torch.from_numpy(
        scaled_targets[val_idx]
    ).float()

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=8,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = FNO2d(
        width=24,
        modes_y=8,
        modes_x=16,
        layers=4,
        padding_y=6,
        padding_x=6,
    ).cpu()

    parameter_count = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=12,
        min_lr=1.0e-5,
    )

    max_epochs = 350
    early_stopping_patience = 45
    min_delta = 1.0e-4

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    history_rows: list[dict[str, float]] = []
    training_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch)
            loss = relative_l2(prediction, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            batch_losses.append(float(loss.detach()))

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_x)
            val_loss = float(
                relative_l2(
                    val_prediction,
                    val_y,
                ).detach()
            )

        scheduler.step(val_loss)
        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "epoch": float(epoch),
                "train_relative_l2_scaled": train_loss,
                "val_relative_l2_scaled": val_loss,
                "learning_rate": learning_rate,
            }
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"train={train_loss:.5f} "
                f"val={val_loss:.5f} "
                f"best={best_val:.5f} "
                f"lr={learning_rate:.2e}",
                flush=True,
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"early stopping at epoch={epoch}, "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    training_seconds = time.perf_counter() - training_start

    if best_state is None:
        raise RuntimeError("Не удалось получить best checkpoint.")

    model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {
            "seed": SEED,
            "model_config": {
                "width": 24,
                "modes_y": 8,
                "modes_x": 16,
                "layers": 4,
                "padding_y": 6,
                "padding_x": 6,
            },
            "model_state": best_state,
            "input_mean": input_mean,
            "input_std": input_std,
            "output_scale": scale,
            "best_epoch": best_epoch,
            "best_val_relative_l2_scaled": best_val,
        },
        args.checkpoint,
    )

    args.history.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with args.history.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                "train_relative_l2_scaled",
                "val_relative_l2_scaled",
                "learning_rate",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history_rows)

    predictions = evaluate_model(
        model,
        raw_inputs,
        displacement,
        input_mean,
        input_std,
        scale,
        test_idx,
    )
    targets = displacement[test_idx]

    field_error = array_relative_l2(
        predictions,
        targets,
    )

    mesh = StructuredQuadMesh(
        nelx=96,
        nely=32,
        length=3.0,
        height=1.0,
    )

    compliance_errors: list[float] = []
    residuals: list[float] = []

    for local_index, sample_index in enumerate(test_idx):
        compliance_error, residual = physical_metrics(
            rho_element[sample_index],
            force[sample_index],
            predictions[local_index],
            float(compliance[sample_index]),
            mesh,
        )
        compliance_errors.append(compliance_error)
        residuals.append(residual)

    compliance_errors_array = np.asarray(
        compliance_errors,
        dtype=float,
    )
    residuals_array = np.asarray(
        residuals,
        dtype=float,
    )

    # Выбираем test sample, ближайший к median field error, для рисунка.
    median_error = float(np.median(field_error))
    figure_local_index = int(
        np.argmin(np.abs(field_error - median_error))
    )
    figure_sample_index = int(
        test_idx[figure_local_index]
    )

    write_comparison_png(
        targets[figure_local_index],
        predictions[figure_local_index],
        args.figure,
    )

    # Точное выполнение граничного условия в выходе FNO.
    boundary_max = float(
        np.max(
            np.abs(
                predictions[:, :, :, 0]
            )
        )
    )

    # Базовый timing на одном median-test sample.
    normalized_one = normalized_inputs[
        figure_sample_index
    ]
    fno_ms = benchmark_fno(
        model,
        normalized_one,
        scale,
        repeats=40,
    )
    fem_ms = benchmark_fem(
        rho_element[figure_sample_index],
        force[figure_sample_index],
        mesh,
        repeats=3,
    )

    # Resolution-transfer: один terminal test condition, 96x32 -> 192x64.
    terminal_test = test_idx[
        snapshot_rank[test_idx] == 5
    ]
    if terminal_test.size == 0:
        raise RuntimeError("Нет terminal test sample.")

    fine_source_index = int(terminal_test[0])
    coarse_vector = rho_element[
        fine_source_index
    ].T.reshape(-1)

    fine_density, fine_nelx, fine_nely = resample_factor_two(
        coarse_vector,
        96,
        32,
        mode="refine",
    )

    fine_mesh = StructuredQuadMesh(
        nelx=fine_nelx,
        nely=fine_nely,
        length=3.0,
        height=1.0,
    )

    q = condition[fine_source_index]
    angle_deg = math.degrees(
        math.atan2(float(q[3]), float(q[2]))
    )

    fine_problem = cantilever_condition(
        fine_mesh,
        volume_fraction=float(q[0]),
        load_y_fraction=float(q[1]),
        load_angle_deg=angle_deg,
        load_magnitude=1.0,
    )

    fine_stiffness = assemble_stiffness(
        fine_mesh,
        young=1.0,
        poisson=0.3,
        thickness=1.0,
        element_factors=simp_factors(fine_density),
    )
    fine_exact = solve_elasticity(
        fine_stiffness,
        fine_problem,
    )

    fine_rho_node = element_to_node_density(
        fine_density,
        fine_mesh,
    )
    fine_force = force_to_grid(
        fine_problem.force,
        fine_mesh,
    )

    fine_raw = build_raw_inputs(
        fine_rho_node[None, ...],
        fine_force[None, ...],
        q[None, ...],
    )
    fine_normalized = (
        fine_raw - input_mean
    ) / input_std

    model.eval()
    with torch.no_grad():
        fine_pred_scaled = model(
            torch.from_numpy(fine_normalized).float()
        )
        fine_prediction = (
            fine_pred_scaled.cpu().numpy()
            * scale
        )[0]

    exact_nodes = fine_exact.displacement.reshape(
        fine_mesh.nnodes,
        2,
    )
    exact_grid = exact_nodes.reshape(
        fine_mesh.nelx + 1,
        fine_mesh.nely + 1,
        2,
    )
    exact_grid = np.transpose(
        exact_grid,
        (2, 1, 0),
    ).astype(np.float32)

    fine_field_error = float(
        array_relative_l2(
            fine_prediction[None, ...],
            exact_grid[None, ...],
        )[0]
    )
    fine_compliance_error, fine_residual = physical_metrics(
        fine_density.reshape(
            fine_mesh.nelx,
            fine_mesh.nely,
        ).T,
        fine_force,
        fine_prediction,
        float(fine_exact.compliance),
        fine_mesh,
    )

    result = {
        "stage": "V05",
        "seed": SEED,
        "device": "cpu",
        "torch_version": torch.__version__,
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "architecture": {
            "name": "FNO2d",
            "input_channels": 9,
            "output_channels": 2,
            "width": 24,
            "modes_y": 8,
            "modes_x": 16,
            "layers": 4,
            "padding_y": 6,
            "padding_x": 6,
            "real_parameter_count": parameter_count,
        },
        "training": {
            "train_samples": int(train_idx.size),
            "val_samples": int(val_idx.size),
            "test_samples": int(test_idx.size),
            "batch_size": 8,
            "optimizer": "AdamW",
            "initial_learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "epochs_run": len(history_rows),
            "best_epoch": best_epoch,
            "best_val_relative_l2_scaled": best_val,
            "training_seconds": training_seconds,
            "normalization_from_train_only": True,
        },
        "test": {
            "field_relative_l2": percentile_summary(
                field_error
            ),
            "compliance_relative_error": percentile_summary(
                compliance_errors_array
            ),
            "physics_residual": percentile_summary(
                residuals_array
            ),
            "max_fixed_boundary_abs_displacement": boundary_max,
            "figure_sample_index": figure_sample_index,
            "figure_condition_id": int(
                condition_id[figure_sample_index]
            ),
            "figure_snapshot_rank": int(
                snapshot_rank[figure_sample_index]
            ),
        },
        "timing": {
            "single_sample_fno_ms": fno_ms,
            "single_sample_fem_ms": fem_ms,
            "fem_over_fno_ratio": fem_ms / max(fno_ms, 1.0e-12),
            "note": (
                "CPU wall-clock microbenchmark on one 96x32 sample; "
                "not a hardware-independent complexity result."
            ),
        },
        "resolution_transfer_192x64": {
            "source_sample_index": fine_source_index,
            "source_condition_id": int(
                condition_id[fine_source_index]
            ),
            "field_relative_l2": fine_field_error,
            "compliance_relative_error": fine_compliance_error,
            "physics_residual": fine_residual,
            "exact_fem_residual": float(
                fine_exact.relative_residual
            ),
            "interpretation": (
                "Exploratory zero-shot evaluation of the same trained "
                "spectral operator on a 192x64 element mesh. "
                "No fine-resolution sample is used for training."
            ),
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

    print("V05 FNO finished")
    print(
        f"params={parameter_count}, "
        f"epochs={len(history_rows)}, "
        f"best_epoch={best_epoch}, "
        f"best_val={best_val:.5f}"
    )
    print(
        "test field rel L2: "
        f"median={np.median(field_error):.5f}, "
        f"p90={np.percentile(field_error, 90):.5f}, "
        f"max={np.max(field_error):.5f}"
    )
    print(
        "test compliance rel error: "
        f"median={np.median(compliance_errors_array):.5f}, "
        f"p90={np.percentile(compliance_errors_array, 90):.5f}"
    )
    print(
        "test physics residual: "
        f"median={np.median(residuals_array):.3e}, "
        f"p90={np.percentile(residuals_array, 90):.3e}"
    )
    print(
        f"timing FNO={fno_ms:.3f} ms, "
        f"FEM={fem_ms:.3f} ms, "
        f"ratio={fem_ms / max(fno_ms, 1e-12):.2f}x"
    )
    print(
        "192x64 zero-shot: "
        f"field={fine_field_error:.5f}, "
        f"compliance={fine_compliance_error:.5f}, "
        f"residual={fine_residual:.3e}"
    )


if __name__ == "__main__":
    main()
