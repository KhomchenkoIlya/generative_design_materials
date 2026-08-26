"""V07: conditional DDPM on downsampled SIMP trajectories."""

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

from project3_diffusion.diffusion import (  # noqa: E402
    ConditionalUNet,
    DiffusionSchedule,
    binarity_score,
    condition_matrix,
    condition_statistics,
    count_parameters,
    downsample_density_2x,
    volume_project,
)


SEED = 20260814
TRAIN_SPLIT = 0
VAL_SPLIT = 1
TEST_SPLIT = 2
CANDIDATES_PER_CONDITION = 16


def set_reproducible_cpu(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(),
            model.parameters(),
        ):
            ema_parameter.mul_(decay).add_(
                parameter,
                alpha=1.0 - decay,
            )


def fixed_noise_loss(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    condition: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        xt = schedule.q_sample(
            x0,
            timesteps,
            noise,
        )
        prediction = model(
            xt,
            timesteps,
            condition,
        )
        loss = torch.mean(
            (prediction - noise) ** 2
        )
    return float(loss)


def pairwise_diversity(
    candidates: np.ndarray,
) -> float:
    candidates = np.asarray(candidates, dtype=float)
    if candidates.shape[0] < 2:
        return 0.0

    distances = []
    norm = math.sqrt(
        candidates.shape[1]
        * candidates.shape[2]
    )

    for i in range(candidates.shape[0]):
        for j in range(i + 1, candidates.shape[0]):
            distances.append(
                np.linalg.norm(
                    candidates[i] - candidates[j]
                )
                / norm
            )

    return float(np.mean(distances))


def nearest_terminal_distance(
    candidates: np.ndarray,
    terminal_train: np.ndarray,
) -> np.ndarray:
    """RMS distance до ближайшей terminal train topology."""

    candidates = np.asarray(candidates, dtype=float)
    terminal_train = np.asarray(
        terminal_train,
        dtype=float,
    )

    output = []
    for candidate in candidates:
        distance = np.sqrt(
            np.mean(
                (
                    terminal_train
                    - candidate[None, ...]
                )
                ** 2,
                axis=(1, 2),
            )
        )
        output.append(float(np.min(distance)))
    return np.asarray(output, dtype=float)


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_montage(
    candidates: np.ndarray,
    path: Path,
    *,
    rows: int,
    columns: int = 4,
    scale: int = 4,
    gap: int = 3,
) -> None:
    """Rows=test conditions, columns=first candidates."""

    candidates = np.asarray(candidates, dtype=float)
    _, per_condition, height, width = candidates.shape

    if per_condition < columns:
        raise ValueError("слишком мало candidates для montage.")

    tile_h = height * scale
    tile_w = width * scale

    canvas_h = rows * tile_h + (rows - 1) * gap
    canvas_w = columns * tile_w + (columns - 1) * gap

    canvas = np.full(
        (canvas_h, canvas_w),
        255,
        dtype=np.uint8,
    )

    for row in range(rows):
        for column in range(columns):
            field = np.clip(
                candidates[row, column],
                0.0,
                1.0,
            )
            gray = np.rint(
                255.0 * (1.0 - field)
            ).astype(np.uint8)
            gray = np.flipud(gray)
            tile = np.repeat(
                np.repeat(gray, scale, axis=0),
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--condition-csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    set_reproducible_cpu(SEED)

    with np.load(
        args.dataset,
        allow_pickle=False,
    ) as archive:
        rho_element = archive["rho_element"].astype(
            np.float32
        )
        condition = archive["condition"].astype(
            np.float32
        )
        split = archive["split"].astype(np.uint8)
        condition_id = archive["condition_id"].astype(
            np.int16
        )
        snapshot_rank = archive["snapshot_rank"].astype(
            np.uint8
        )
        progress = archive["progress"].astype(
            np.float32
        )

    rho_small = downsample_density_2x(
        rho_element
    )
    x0 = (
        2.0 * rho_small - 1.0
    ).astype(np.float32)

    all_condition = condition_matrix(
        condition,
        progress,
    )

    train_mask = split == TRAIN_SPLIT
    val_mask = split == VAL_SPLIT
    test_mask = split == TEST_SPLIT

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    condition_mean, condition_std = condition_statistics(
        all_condition,
        train_mask,
    )
    normalized_condition = (
        all_condition - condition_mean
    ) / condition_std

    train_x = torch.from_numpy(
        x0[train_idx, None, ...]
    ).float()
    train_c = torch.from_numpy(
        normalized_condition[train_idx]
    ).float()

    val_x = torch.from_numpy(
        x0[val_idx, None, ...]
    ).float()
    val_c = torch.from_numpy(
        normalized_condition[val_idx]
    ).float()

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    train_loader = DataLoader(
        TensorDataset(train_x, train_c),
        batch_size=12,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    model = ConditionalUNet(
        base_channels=32,
        embedding_dim=64,
        condition_dim=5,
    ).cpu()
    ema_model = copy.deepcopy(model).eval()

    parameter_count = count_parameters(model)

    schedule = DiffusionSchedule(
        steps=100,
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        device="cpu",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-4,
        weight_decay=1.0e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=1.0e-5,
    )

    fixed_generator = torch.Generator()
    fixed_generator.manual_seed(SEED + 1)

    val_t = torch.randint(
        0,
        schedule.steps,
        (val_x.shape[0],),
        generator=fixed_generator,
        dtype=torch.long,
    )
    val_noise = torch.randn(
        val_x.shape,
        generator=fixed_generator,
    )

    max_epochs = 300
    early_stopping_patience = 55
    min_delta = 2.0e-4
    ema_decay = 0.995

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, float]] = []

    training_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []

        for batch_x0, batch_c in train_loader:
            batch = batch_x0.shape[0]

            t = torch.randint(
                0,
                schedule.steps,
                (batch,),
                dtype=torch.long,
            )
            noise = torch.randn_like(batch_x0)

            xt = schedule.q_sample(
                batch_x0,
                t,
                noise,
            )
            prediction = model(
                xt,
                t,
                batch_c,
            )

            loss = torch.mean(
                (prediction - noise) ** 2
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            update_ema(
                ema_model,
                model,
                ema_decay,
            )

            losses.append(float(loss.detach()))

        train_loss = float(np.mean(losses))
        val_loss = fixed_noise_loss(
            ema_model,
            schedule,
            val_x,
            val_c,
            val_t,
            val_noise,
        )

        scheduler.step(val_loss)
        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_noise_mse": train_loss,
                "val_noise_mse": val_loss,
                "learning_rate": learning_rate,
            }
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                ema_model.state_dict()
            )
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

    training_seconds = (
        time.perf_counter() - training_start
    )

    if best_state is None:
        raise RuntimeError("Не получен best DDPM checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    args.checkpoint.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(
        {
            "seed": SEED,
            "model_config": {
                "base_channels": 32,
                "embedding_dim": 64,
                "condition_dim": 5,
            },
            "model_state": best_state,
            "condition_mean": condition_mean,
            "condition_std": condition_std,
            "diffusion": {
                "steps": 100,
                "beta_start": 1.0e-4,
                "beta_end": 2.0e-2,
            },
            "best_epoch": best_epoch,
            "best_val_noise_mse": best_val,
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
                "train_noise_mse",
                "val_noise_mse",
                "learning_rate",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history_rows)

    # Test noise-prediction loss is diagnostic only; it does not tune the model.
    test_x = torch.from_numpy(
        x0[test_idx, None, ...]
    ).float()
    test_c = torch.from_numpy(
        normalized_condition[test_idx]
    ).float()

    test_generator = torch.Generator()
    test_generator.manual_seed(SEED + 2)

    test_t = torch.randint(
        0,
        schedule.steps,
        (test_x.shape[0],),
        generator=test_generator,
        dtype=torch.long,
    )
    test_noise = torch.randn(
        test_x.shape,
        generator=test_generator,
    )

    test_noise_mse = fixed_noise_loss(
        model,
        schedule,
        test_x,
        test_c,
        test_t,
        test_noise,
    )

    # Один engineering condition = один condition_id.
    unique_test_ids = sorted(
        int(value)
        for value in np.unique(
            condition_id[test_idx]
        )
    )

    raw_all: list[np.ndarray] = []
    projected_all: list[np.ndarray] = []
    generated_condition: list[np.ndarray] = []
    generated_ids: list[int] = []
    condition_rows: list[dict[str, object]] = []

    terminal_train = rho_small[
        train_mask & (snapshot_rank == 5)
    ]
    terminal_test = rho_small[
        test_mask & (snapshot_rank == 5)
    ]

    raw_volume_errors_all: list[float] = []
    projected_volume_errors_all: list[float] = []
    binarity_all: list[float] = []
    novelty_all: list[float] = []
    diversity_by_condition: list[float] = []

    for ordinal, cid in enumerate(unique_test_ids):
        indices = np.flatnonzero(
            (split == TEST_SPLIT)
            & (condition_id == cid)
        )
        terminal_indices = indices[
            snapshot_rank[indices] == 5
        ]
        if terminal_indices.size != 1:
            raise RuntimeError(
                f"condition {cid}: нужен ровно один terminal sample."
            )

        source_index = int(terminal_indices[0])
        engineering_condition = condition[source_index]
        target_volume = float(
            engineering_condition[0]
        )

        sampling_condition = np.concatenate(
            (
                engineering_condition,
                np.array([1.0], dtype=np.float32),
            )
        )[None, :]
        sampling_condition = np.repeat(
            sampling_condition,
            CANDIDATES_PER_CONDITION,
            axis=0,
        )
        sampling_condition = (
            sampling_condition - condition_mean
        ) / condition_std

        sampling_tensor = torch.from_numpy(
            sampling_condition.astype(np.float32)
        )

        generator = torch.Generator()
        generator.manual_seed(
            SEED + 100 + ordinal
        )

        sampled = schedule.sample(
            model,
            sampling_tensor,
            generator=generator,
        )
        raw = (
            0.5
            * (
                sampled[:, 0].cpu().numpy()
                + 1.0
            )
        ).astype(np.float32)

        projected = np.stack(
            [
                volume_project(
                    candidate,
                    target_volume,
                )
                for candidate in raw
            ],
            axis=0,
        )

        raw_volume_errors = np.abs(
            raw.mean(axis=(1, 2))
            - target_volume
        )
        projected_volume_errors = np.abs(
            projected.mean(axis=(1, 2))
            - target_volume
        )

        candidate_binarity = np.asarray(
            [
                binarity_score(candidate)
                for candidate in projected
            ],
            dtype=float,
        )

        novelty = nearest_terminal_distance(
            projected,
            terminal_train,
        )
        diversity = pairwise_diversity(
            projected
        )

        angle = math.degrees(
            math.atan2(
                float(engineering_condition[3]),
                float(engineering_condition[2]),
            )
        )

        condition_rows.append(
            {
                "condition_id": cid,
                "target_volume": target_volume,
                "load_y_fraction": float(
                    engineering_condition[1]
                ),
                "load_angle_deg": angle,
                "candidate_count": CANDIDATES_PER_CONDITION,
                "raw_volume_error_median": float(
                    np.median(raw_volume_errors)
                ),
                "raw_volume_error_p90": float(
                    np.percentile(
                        raw_volume_errors,
                        90,
                    )
                ),
                "projected_volume_error_max": float(
                    np.max(projected_volume_errors)
                ),
                "pairwise_diversity": diversity,
                "projected_binarity_median": float(
                    np.median(candidate_binarity)
                ),
                "nearest_train_terminal_rms_median": float(
                    np.median(novelty)
                ),
            }
        )

        raw_volume_errors_all.extend(
            raw_volume_errors.tolist()
        )
        projected_volume_errors_all.extend(
            projected_volume_errors.tolist()
        )
        binarity_all.extend(
            candidate_binarity.tolist()
        )
        novelty_all.extend(
            novelty.tolist()
        )
        diversity_by_condition.append(diversity)

        raw_all.append(raw)
        projected_all.append(projected)
        generated_condition.append(
            engineering_condition.copy()
        )
        generated_ids.append(cid)

    raw_array = np.stack(raw_all, axis=0)
    projected_array = np.stack(
        projected_all,
        axis=0,
    )

    args.candidates.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        args.candidates,
        raw_density=raw_array,
        projected_density=projected_array,
        condition=np.stack(
            generated_condition,
            axis=0,
        ).astype(np.float32),
        condition_id=np.asarray(
            generated_ids,
            dtype=np.int16,
        ),
        candidate_seed=np.asarray(
            [
                SEED + 100 + i
                for i in range(len(unique_test_ids))
            ],
            dtype=np.int64,
        ),
    )

    args.condition_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with args.condition_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "condition_id",
                "target_volume",
                "load_y_fraction",
                "load_angle_deg",
                "candidate_count",
                "raw_volume_error_median",
                "raw_volume_error_p90",
                "projected_volume_error_max",
                "pairwise_diversity",
                "projected_binarity_median",
                "nearest_train_terminal_rms_median",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(condition_rows)

    write_montage(
        projected_array,
        args.figure,
        rows=len(unique_test_ids),
        columns=4,
    )

    terminal_test_binarity = np.asarray(
        [
            binarity_score(field)
            for field in terminal_test
        ],
        dtype=float,
    )

    result = {
        "stage": "V07",
        "seed": SEED,
        "device": "cpu",
        "dataset_sha256": sha256_file(args.dataset),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "candidate_npz_sha256": sha256_file(args.candidates),
        "candidate_npz_bytes": args.candidates.stat().st_size,
        "architecture": {
            "name": "conditional_DDPM_UNet",
            "grid_elements": [48, 16],
            "base_channels": 32,
            "embedding_dim": 64,
            "condition_dim": 5,
            "condition_fields": [
                "f_vol",
                "load_y_fraction",
                "cos_load_angle",
                "sin_load_angle",
                "simp_progress",
            ],
            "parameter_count": parameter_count,
        },
        "diffusion": {
            "steps": 100,
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
            "prediction_target": "epsilon",
        },
        "training": {
            "train_samples": int(train_idx.size),
            "val_samples": int(val_idx.size),
            "test_samples": int(test_idx.size),
            "batch_size": 12,
            "optimizer": "AdamW",
            "initial_learning_rate": 2.0e-4,
            "weight_decay": 1.0e-4,
            "ema_decay": ema_decay,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "epochs_run": len(history_rows),
            "best_epoch": best_epoch,
            "best_val_noise_mse": best_val,
            "test_noise_mse": test_noise_mse,
            "training_seconds": training_seconds,
            "condition_normalization_train_only": True,
        },
        "sampling": {
            "unseen_test_conditions": len(unique_test_ids),
            "candidates_per_condition": CANDIDATES_PER_CONDITION,
            "candidate_count": int(
                len(unique_test_ids)
                * CANDIDATES_PER_CONDITION
            ),
            "sampling_progress": 1.0,
            "raw_volume_absolute_error": summary(
                np.asarray(
                    raw_volume_errors_all,
                    dtype=float,
                )
            ),
            "projected_volume_absolute_error": summary(
                np.asarray(
                    projected_volume_errors_all,
                    dtype=float,
                )
            ),
            "pairwise_diversity_by_condition": summary(
                np.asarray(
                    diversity_by_condition,
                    dtype=float,
                )
            ),
            "projected_binarity": summary(
                np.asarray(
                    binarity_all,
                    dtype=float,
                )
            ),
            "terminal_test_binarity_reference": summary(
                terminal_test_binarity
            ),
            "nearest_train_terminal_rms": summary(
                np.asarray(
                    novelty_all,
                    dtype=float,
                )
            ),
        },
        "condition_metrics": condition_rows,
        "interpretation": {
            "raw_generation_is_not_declared_feasible": True,
            "volume_projection_is_postprocessing": True,
            "fem_not_used_for_candidate_selection_in_v07": True,
            "next_stage": (
                "V08 upsamples projected candidates to 96x32, "
                "checks exact FEM, connectivity/manufacturability, "
                "and performs a small number of SIMP refinement steps."
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

    print("V07 conditional diffusion finished")
    print(
        f"params={parameter_count}, "
        f"epochs={len(history_rows)}, "
        f"best_epoch={best_epoch}, "
        f"best_val_noise_mse={best_val:.5f}, "
        f"test_noise_mse={test_noise_mse:.5f}"
    )
    print(
        "raw volume abs error: "
        f"median={np.median(raw_volume_errors_all):.5f}, "
        f"p90={np.percentile(raw_volume_errors_all, 90):.5f}"
    )
    print(
        "projected volume max error: "
        f"{np.max(projected_volume_errors_all):.3e}"
    )
    print(
        "diversity by condition: "
        f"median={np.median(diversity_by_condition):.5f}"
    )
    print(
        "projected binarity: "
        f"median={np.median(binarity_all):.5f}; "
        "terminal test reference: "
        f"median={np.median(terminal_test_binarity):.5f}"
    )
    print(
        "nearest train terminal RMS: "
        f"median={np.median(novelty_all):.5f}"
    )


if __name__ == "__main__":
    main()
