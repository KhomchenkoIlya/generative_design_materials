"""V06: анализ ошибок уже обученного FNO без повторного обучения."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import StructuredQuadMesh  # noqa: E402
from project2_operator.fno import FNO2d, build_raw_inputs  # noqa: E402
from project2_operator.train_fno import (  # noqa: E402
    array_relative_l2,
    inverse_grid,
    physical_metrics,
)


TEST_SPLIT = 2
STAGE_FRACTIONS = (0.0, 0.05, 0.15, 0.35, 0.65, 1.0)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def dense_ranks(values: np.ndarray) -> np.ndarray:
    """Уникальные значения -> ранги 0..n-1 в порядке возрастания."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    return ranks


def spearman_no_ties(x: np.ndarray, y: np.ndarray) -> float:
    rx = dense_ranks(np.asarray(x, dtype=float))
    ry = dense_ranks(np.asarray(y, dtype=float))
    if rx.size < 2:
        return 1.0
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_order_accuracy(
    exact: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Доля пар кандидатов, для которых surrogate сохраняет порядок J."""

    exact = np.asarray(exact, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    correct = 0
    total = 0

    for i in range(exact.size):
        for j in range(i + 1, exact.size):
            exact_diff = exact[i] - exact[j]
            pred_diff = predicted[i] - predicted[j]

            if exact_diff == 0.0:
                continue

            total += 1
            if np.sign(exact_diff) == np.sign(pred_diff):
                correct += 1

    if total == 0:
        return 1.0

    return correct / total


def load_model(
    checkpoint_path: Path,
) -> tuple[FNO2d, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["model_config"]
    model = FNO2d(**config).cpu()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_mean = np.asarray(
        checkpoint["input_mean"],
        dtype=np.float32,
    )
    input_std = np.asarray(
        checkpoint["input_std"],
        dtype=np.float32,
    )
    output_scale = np.asarray(
        checkpoint["output_scale"],
        dtype=np.float32,
    )

    return model, input_mean, input_std, output_scale


def predict_test(
    model: FNO2d,
    raw_inputs: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_scale: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    normalized = (
        raw_inputs[test_idx] - input_mean
    ) / input_std

    with torch.no_grad():
        prediction_scaled = model(
            torch.from_numpy(normalized).float()
        )
        prediction = (
            prediction_scaled.cpu().numpy()
            * output_scale
        )

    return prediction.astype(np.float32)


def predicted_compliance(
    force_grid: np.ndarray,
    displacement_grid: np.ndarray,
) -> float:
    return float(
        inverse_grid(force_grid)
        @ inverse_grid(displacement_grid)
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v05-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(
        max(1, min(4, torch.get_num_threads()))
    )

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
        progress = archive["progress"].astype(np.float32)
        iteration = archive["iteration"].astype(np.int16)

    test_idx = np.flatnonzero(split == TEST_SPLIT)

    raw_inputs = build_raw_inputs(
        rho_node,
        force,
        condition,
    )

    model, input_mean, input_std, output_scale = load_model(
        args.checkpoint
    )

    prediction = predict_test(
        model,
        raw_inputs,
        input_mean,
        input_std,
        output_scale,
        test_idx,
    )
    target = displacement[test_idx]

    field_error = array_relative_l2(
        prediction,
        target,
    )

    mesh = StructuredQuadMesh(
        nelx=96,
        nely=32,
        length=3.0,
        height=1.0,
    )

    compliance_error: list[float] = []
    residual: list[float] = []
    predicted_j: list[float] = []

    sample_rows: list[dict[str, object]] = []

    for local_index, sample_index in enumerate(test_idx):
        e_j, r_phys = physical_metrics(
            rho_element[sample_index],
            force[sample_index],
            prediction[local_index],
            float(compliance[sample_index]),
            mesh,
        )

        j_pred = predicted_compliance(
            force[sample_index],
            prediction[local_index],
        )

        compliance_error.append(e_j)
        residual.append(r_phys)
        predicted_j.append(j_pred)

        q = condition[sample_index]
        angle = math.degrees(
            math.atan2(float(q[3]), float(q[2]))
        )

        sample_rows.append(
            {
                "sample_index": int(sample_index),
                "condition_id": int(condition_id[sample_index]),
                "snapshot_rank": int(snapshot_rank[sample_index]),
                "progress": float(progress[sample_index]),
                "iteration": int(iteration[sample_index]),
                "target_volume": float(q[0]),
                "load_y_fraction": float(q[1]),
                "load_angle_deg": angle,
                "field_relative_l2": float(field_error[local_index]),
                "exact_compliance": float(compliance[sample_index]),
                "predicted_compliance": j_pred,
                "compliance_relative_error": e_j,
                "physics_residual": r_phys,
            }
        )

    compliance_error_array = np.asarray(
        compliance_error,
        dtype=float,
    )
    residual_array = np.asarray(
        residual,
        dtype=float,
    )
    predicted_j_array = np.asarray(
        predicted_j,
        dtype=float,
    )

    stage_rows: list[dict[str, object]] = []
    stage_json: dict[str, object] = {}

    for rank, fraction in enumerate(STAGE_FRACTIONS):
        mask = snapshot_rank[test_idx] == rank

        field_summary = percentile_summary(
            field_error[mask]
        )
        compliance_summary = percentile_summary(
            compliance_error_array[mask]
        )
        residual_summary = percentile_summary(
            residual_array[mask]
        )

        key = str(rank)
        stage_json[key] = {
            "progress_fraction": fraction,
            "sample_count": int(np.count_nonzero(mask)),
            "field_relative_l2": field_summary,
            "compliance_relative_error": compliance_summary,
            "physics_residual": residual_summary,
        }

        stage_rows.append(
            {
                "snapshot_rank": rank,
                "progress_fraction": fraction,
                "sample_count": int(np.count_nonzero(mask)),
                "field_median": field_summary["median"],
                "field_p90": field_summary["p90"],
                "compliance_median": compliance_summary["median"],
                "compliance_p90": compliance_summary["p90"],
                "residual_median": residual_summary["median"],
                "residual_p90": residual_summary["p90"],
            }
        )

    condition_rows: list[dict[str, object]] = []
    condition_json: dict[str, object] = {}

    exact_all_pairs: list[float] = []
    predicted_all_pairs: list[float] = []
    best_stage_matches = 0
    improvement_sign_matches = 0

    for cid in sorted(np.unique(condition_id[test_idx])):
        mask = condition_id[test_idx] == cid
        local_indices = np.flatnonzero(mask)
        sample_indices = test_idx[mask]

        order = np.argsort(
            snapshot_rank[sample_indices]
        )
        local_indices = local_indices[order]
        sample_indices = sample_indices[order]

        exact = compliance[sample_indices]
        pred = predicted_j_array[local_indices]

        pair_accuracy = pairwise_order_accuracy(
            exact,
            pred,
        )
        rank_corr = spearman_no_ties(
            exact,
            pred,
        )

        exact_best_rank = int(
            snapshot_rank[
                sample_indices[int(np.argmin(exact))]
            ]
        )
        predicted_best_rank = int(
            snapshot_rank[
                sample_indices[int(np.argmin(pred))]
            ]
        )

        best_match = exact_best_rank == predicted_best_rank
        best_stage_matches += int(best_match)

        exact_improvement = exact[-1] - exact[0]
        predicted_improvement = pred[-1] - pred[0]
        improvement_match = (
            np.sign(exact_improvement)
            == np.sign(predicted_improvement)
        )
        improvement_sign_matches += int(improvement_match)

        for i in range(exact.size):
            for j in range(i + 1, exact.size):
                exact_all_pairs.append(float(exact[i] - exact[j]))
                predicted_all_pairs.append(float(pred[i] - pred[j]))

        q = condition[sample_indices[0]]
        angle = math.degrees(
            math.atan2(float(q[3]), float(q[2]))
        )

        field_summary = percentile_summary(
            field_error[local_indices]
        )
        compliance_summary = percentile_summary(
            compliance_error_array[local_indices]
        )
        residual_summary = percentile_summary(
            residual_array[local_indices]
        )

        key = str(int(cid))
        condition_json[key] = {
            "target_volume": float(q[0]),
            "load_y_fraction": float(q[1]),
            "load_angle_deg": angle,
            "field_relative_l2": field_summary,
            "compliance_relative_error": compliance_summary,
            "physics_residual": residual_summary,
            "pairwise_compliance_order_accuracy": pair_accuracy,
            "spearman_compliance": rank_corr,
            "exact_best_snapshot_rank": exact_best_rank,
            "predicted_best_snapshot_rank": predicted_best_rank,
            "best_snapshot_match": best_match,
            "initial_to_terminal_improvement_sign_match": bool(
                improvement_match
            ),
        }

        condition_rows.append(
            {
                "condition_id": int(cid),
                "target_volume": float(q[0]),
                "load_y_fraction": float(q[1]),
                "load_angle_deg": angle,
                "field_median": field_summary["median"],
                "compliance_median": compliance_summary["median"],
                "residual_median": residual_summary["median"],
                "pairwise_order_accuracy": pair_accuracy,
                "spearman_compliance": rank_corr,
                "exact_best_snapshot_rank": exact_best_rank,
                "predicted_best_snapshot_rank": predicted_best_rank,
            }
        )

    overall_pairwise = pairwise_order_accuracy(
        np.asarray(exact_all_pairs),
        np.asarray(predicted_all_pairs),
    )

    # pairwise_order_accuracy ожидает сами значения, а выше лежат pair-differences.
    # Для объединённой оценки считаем знаки напрямую.
    exact_diffs = np.asarray(exact_all_pairs, dtype=float)
    pred_diffs = np.asarray(predicted_all_pairs, dtype=float)
    overall_pairwise = float(
        np.mean(np.sign(exact_diffs) == np.sign(pred_diffs))
    )

    field_vs_compliance = float(
        np.corrcoef(
            field_error,
            compliance_error_array,
        )[0, 1]
    )
    field_vs_log_residual = float(
        np.corrcoef(
            field_error,
            np.log10(residual_array),
        )[0, 1]
    )

    worst_stage = int(
        max(
            stage_json,
            key=lambda key: stage_json[key][
                "field_relative_l2"
            ]["median"],
        )
    )
    best_stage = int(
        min(
            stage_json,
            key=lambda key: stage_json[key][
                "field_relative_l2"
            ]["median"],
        )
    )

    v05 = json.loads(
        args.v05_result.read_text(encoding="utf-8")
    )

    result = {
        "stage": "V06",
        "analysis_uses_fixed_v05_checkpoint": True,
        "test_sample_count": int(test_idx.size),
        "test_condition_count": int(
            np.unique(condition_id[test_idx]).size
        ),
        "stage_metrics": stage_json,
        "condition_metrics": condition_json,
        "screening": {
            "within_condition_pairwise_compliance_order_accuracy": (
                overall_pairwise
            ),
            "best_snapshot_matches": best_stage_matches,
            "best_snapshot_total": int(
                np.unique(condition_id[test_idx]).size
            ),
            "initial_to_terminal_improvement_sign_matches": (
                improvement_sign_matches
            ),
            "initial_to_terminal_total": int(
                np.unique(condition_id[test_idx]).size
            ),
        },
        "relationships": {
            "pearson_field_vs_compliance_error": (
                field_vs_compliance
            ),
            "pearson_field_vs_log10_physics_residual": (
                field_vs_log_residual
            ),
        },
        "stage_extremes": {
            "lowest_median_field_error_snapshot_rank": best_stage,
            "highest_median_field_error_snapshot_rank": worst_stage,
        },
        "v05_reference": {
            "field_median": v05["test"][
                "field_relative_l2"
            ]["median"],
            "field_p90": v05["test"][
                "field_relative_l2"
            ]["p90"],
            "compliance_median": v05["test"][
                "compliance_relative_error"
            ]["median"],
            "physics_residual_median": v05["test"][
                "physics_residual"
            ]["median"],
            "fem_over_fno_ratio": v05["timing"][
                "fem_over_fno_ratio"
            ],
            "resolution_transfer_field_error": v05[
                "resolution_transfer_192x64"
            ]["field_relative_l2"],
        },
        "interpretation_guardrails": {
            "parameter_effects_are_not_causal": True,
            "reason": (
                "Only five held-out test conditions are available and "
                "the factor levels are not balanced across this test subset. "
                "Condition-wise differences are descriptive, not isolated "
                "effects of volume, load position or angle."
            ),
            "physics_statement": (
                "Small displacement L2 error does not imply small "
                "equilibrium residual because K acts on the displacement "
                "error: K(Uhat-U)=K Uhat-F on free DOFs."
            ),
        },
    }

    args.result.parent.mkdir(parents=True, exist_ok=True)
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

    write_csv(
        args.samples,
        sample_rows,
        (
            "sample_index",
            "condition_id",
            "snapshot_rank",
            "progress",
            "iteration",
            "target_volume",
            "load_y_fraction",
            "load_angle_deg",
            "field_relative_l2",
            "exact_compliance",
            "predicted_compliance",
            "compliance_relative_error",
            "physics_residual",
        ),
    )

    write_csv(
        args.stages,
        stage_rows,
        (
            "snapshot_rank",
            "progress_fraction",
            "sample_count",
            "field_median",
            "field_p90",
            "compliance_median",
            "compliance_p90",
            "residual_median",
            "residual_p90",
        ),
    )

    write_csv(
        args.conditions,
        condition_rows,
        (
            "condition_id",
            "target_volume",
            "load_y_fraction",
            "load_angle_deg",
            "field_median",
            "compliance_median",
            "residual_median",
            "pairwise_order_accuracy",
            "spearman_compliance",
            "exact_best_snapshot_rank",
            "predicted_best_snapshot_rank",
        ),
    )

    print("V06 FNO analysis finished")
    print(
        "pairwise compliance ordering="
        f"{overall_pairwise:.3f}"
    )
    print(
        "best snapshot matches="
        f"{best_stage_matches}/"
        f"{result['screening']['best_snapshot_total']}"
    )
    print(
        "initial->terminal improvement sign="
        f"{improvement_sign_matches}/"
        f"{result['screening']['initial_to_terminal_total']}"
    )
    print(
        "corr(field, compliance error)="
        f"{field_vs_compliance:.3f}"
    )
    print(
        "corr(field, log10 residual)="
        f"{field_vs_log_residual:.3f}"
    )
    print(
        f"best stage={best_stage}, "
        f"worst stage={worst_stage}"
    )


if __name__ == "__main__":
    main()
