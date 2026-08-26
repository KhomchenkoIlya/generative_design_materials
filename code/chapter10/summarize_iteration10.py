"""V09: cross-project summary for HiddenPower iteration 10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentage(value: float) -> float:
    return 100.0 * float(value)


def fno_role(
    *,
    speedup: float,
    pairwise_ordering: float,
    physics_residual: float,
) -> str:
    if (
        speedup > 1.0
        and pairwise_ordering >= 0.9
        and physics_residual > 1.0
    ):
        return "screening_surrogate_not_final_solver"
    return "insufficient_evidence_for_screening"


def diffusion_role(
    *,
    connected_fraction: float,
    warmstart_ratio: float,
    rms_change: float,
) -> str:
    if (
        connected_fraction < 0.5
        and abs(warmstart_ratio - 1.0) < 0.02
        and rms_change > 0.2
    ):
        return "candidate_generator_with_modest_warmstart_effect"
    return "different_regime"


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fieldnames = (
        "project",
        "method",
        "primary_role",
        "exact_physics_in_final_decision",
        "headline_metric",
        "headline_value",
        "limitation",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--v03", type=Path, required=True)
    parser.add_argument("--v05", type=Path, required=True)
    parser.add_argument("--v06", type=Path, required=True)
    parser.add_argument("--v07", type=Path, required=True)
    parser.add_argument("--v08", type=Path, required=True)

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fno-checkpoint", type=Path, required=True)
    parser.add_argument("--ddpm-checkpoint", type=Path, required=True)
    parser.add_argument("--ddpm-candidates", type=Path, required=True)
    parser.add_argument("--v08-refined", type=Path, required=True)

    parser.add_argument("--git-head", type=str, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)

    args = parser.parse_args()

    v03 = load_json(args.v03)
    v05 = load_json(args.v05)
    v06 = load_json(args.v06)
    v07 = load_json(args.v07)
    v08 = load_json(args.v08)

    assert v03["stage"] == "V03"
    assert v05["stage"] == "V05"
    assert v06["stage"] == "V06"
    assert v07["stage"] == "V07"
    assert v08["stage"] == "V08"

    cantilever = v03["cases"]["cantilever"]

    simp_initial = float(cantilever["initial_compliance"])
    simp_final = float(cantilever["final_compliance"])
    simp_iterations = int(cantilever["iterations"])
    simp_runtime = float(cantilever["runtime_seconds"])
    simp_reduction = 1.0 - simp_final / simp_initial

    fno_field = float(
        v05["test"]["field_relative_l2"]["median"]
    )
    fno_compliance = float(
        v05["test"]["compliance_relative_error"]["median"]
    )
    fno_residual = float(
        v05["test"]["physics_residual"]["median"]
    )
    fno_speedup = float(
        v05["timing"]["fem_over_fno_ratio"]
    )
    fno_resolution_error = float(
        v05["resolution_transfer_192x64"][
            "field_relative_l2"
        ]
    )

    pairwise = float(
        v06["screening"][
            "within_condition_pairwise_compliance_order_accuracy"
        ]
    )
    best_snapshot_matches = int(
        v06["screening"]["best_snapshot_matches"]
    )
    best_snapshot_total = int(
        v06["screening"]["best_snapshot_total"]
    )

    ddpm_training_seconds = float(
        v07["training"]["training_seconds"]
    )
    ddpm_raw_volume_median = float(
        v07["sampling"]["raw_volume_absolute_error"]["median"]
    )
    ddpm_binarity = float(
        v07["sampling"]["projected_binarity"]["median"]
    )
    terminal_binarity_v07 = float(
        v07["sampling"][
            "terminal_test_binarity_reference"
        ]["median"]
    )
    ddpm_diversity = float(
        v07["sampling"][
            "pairwise_diversity_by_condition"
        ]["median"]
    )

    screening_connected = int(
        v08["screening"]["connected_at_0_5_count"]
    )
    screening_total = int(
        v08["protocol"]["screen_candidates"]
    )
    screening_connected_fraction = (
        screening_connected / screening_total
    )

    refinement = v08["refinement"]

    diffusion_better_count = int(
        refinement["diffusion_better_uniform30_count"]
    )
    condition_count = int(
        refinement["condition_count"]
    )

    refined_over_uniform = float(
        refinement["best_refined_over_uniform30"]["median"]
    )
    refined_over_terminal = float(
        refinement["best_refined_over_terminal"]["median"]
    )
    uniform_over_terminal = float(
        refinement["uniform30_over_terminal"]["median"]
    )
    refined_over_generated = float(
        refinement[
            "initial_to_refined_compliance_ratio"
        ]["median"]
    )

    rms_change = float(
        refinement["rms_density_change"]["median"]
    )

    selected_connected_before = int(
        refinement["selected_connected_before_count"]
    )
    selected_connected_after = int(
        refinement["selected_connected_after_count"]
    )
    selected_count = int(
        refinement["selected_count"]
    )

    refined_binarity = float(
        refinement["refined_selected_binarity"]["median"]
    )
    terminal_binarity = float(
        refinement["terminal_binarity"]["median"]
    )

    v08_runtime = float(v08["runtime_seconds"])

    role_fno = fno_role(
        speedup=fno_speedup,
        pairwise_ordering=pairwise,
        physics_residual=fno_residual,
    )

    role_diffusion = diffusion_role(
        connected_fraction=screening_connected_fraction,
        warmstart_ratio=refined_over_uniform,
        rms_change=rms_change,
    )

    rows = [
        {
            "project": "1",
            "method": "FEM+SIMP",
            "primary_role": "exact_physics_and_optimization_baseline",
            "exact_physics_in_final_decision": "yes",
            "headline_metric": "cantilever_compliance_reduction",
            "headline_value": f"{percentage(simp_reduction):.2f}%",
            "limitation": "iterative_exact_computation",
        },
        {
            "project": "2",
            "method": "FNO",
            "primary_role": role_fno,
            "exact_physics_in_final_decision": "yes_after_screening",
            "headline_metric": "pairwise_compliance_ordering",
            "headline_value": f"{percentage(pairwise):.1f}%",
            "limitation": "large_equilibrium_residual_and_resolution_gap",
        },
        {
            "project": "3",
            "method": "conditional_DDPM+FEM+SIMP",
            "primary_role": role_diffusion,
            "exact_physics_in_final_decision": "yes",
            "headline_metric": "median_refined_over_uniform30",
            "headline_value": f"{refined_over_uniform:.5f}",
            "limitation": "raw_candidates_gray_and_often_disconnected",
        },
    ]

    summary = {
        "stage": "V09",
        "git_head_before_v09": args.git_head,
        "project1": {
            "method": "FEM+SIMP",
            "cantilever_initial_compliance": simp_initial,
            "cantilever_final_compliance": simp_final,
            "cantilever_compliance_reduction": simp_reduction,
            "cantilever_iterations": simp_iterations,
            "cantilever_runtime_seconds": simp_runtime,
            "conclusion": (
                "Reference route: exact FEM supplies every objective and "
                "SIMP performs the actual optimization."
            ),
        },
        "project2": {
            "method": "FNO surrogate",
            "test_field_relative_l2_median": fno_field,
            "test_compliance_relative_error_median": fno_compliance,
            "test_physics_residual_median": fno_residual,
            "single_sample_fem_over_fno": fno_speedup,
            "training_seconds": float(v05["training"]["training_seconds"]),
            "pairwise_compliance_ordering": pairwise,
            "best_snapshot_matches": best_snapshot_matches,
            "best_snapshot_total": best_snapshot_total,
            "resolution_transfer_192x64_field_error": (
                fno_resolution_error
            ),
            "role": role_fno,
        },
        "project3": {
            "method": "conditional DDPM + exact FEM + fixed-budget SIMP",
            "ddpm_training_seconds": ddpm_training_seconds,
            "raw_volume_error_median": ddpm_raw_volume_median,
            "ddpm_diversity_median": ddpm_diversity,
            "generated_binarity_median": ddpm_binarity,
            "terminal_binarity_reference_v07": (
                terminal_binarity_v07
            ),
            "screen_connected_count": screening_connected,
            "screen_total": screening_total,
            "screen_connected_fraction": (
                screening_connected_fraction
            ),
            "diffusion_better_uniform30_count": (
                diffusion_better_count
            ),
            "condition_count": condition_count,
            "median_refined_over_uniform30": (
                refined_over_uniform
            ),
            "median_refined_over_terminal": (
                refined_over_terminal
            ),
            "median_uniform30_over_terminal": (
                uniform_over_terminal
            ),
            "median_refined_over_generated": (
                refined_over_generated
            ),
            "selected_connected_before": (
                selected_connected_before
            ),
            "selected_connected_after": (
                selected_connected_after
            ),
            "selected_count": selected_count,
            "refined_binarity_median": refined_binarity,
            "terminal_binarity_median": terminal_binarity,
            "median_rms_density_change": rms_change,
            "v08_runtime_seconds": v08_runtime,
            "role": role_diffusion,
        },
        "cross_project_conclusion": {
            "exact_fem_is_final_authority": True,
            "fno_useful_for_screening": (
                role_fno
                == "screening_surrogate_not_final_solver"
            ),
            "fno_is_final_solver": False,
            "diffusion_is_direct_design_solver": False,
            "diffusion_warm_start_effect_is_large": False,
            "diffusion_warm_start_effect_is_mixed_but_nonzero": (
                diffusion_better_count > condition_count / 2
                and abs(refined_over_uniform - 1.0) < 0.02
            ),
            "physics_reentry_is_required_after_ml": True,
        },
        "scope_guardrails": {
            "fno_timing_is_cpu_wall_clock_not_complexity": True,
            "fno_resolution_invariance_not_claimed": True,
            "five_test_conditions_do_not_identify_causal_parameter_effects": True,
            "diffusion_advantage_not_claimed_as_universal": True,
            "all_final_engineering_comparisons_use_exact_fem": True,
        },
    }

    tracked_results = {
        "v03_topopt.json": args.v03,
        "v05_fno_results.json": args.v05,
        "v06_fno_analysis.json": args.v06,
        "v07_diffusion_results.json": args.v07,
        "v08_diffusion_refinement.json": args.v08,
    }

    local_artifacts = {
        "v04_operator_dataset.npz": args.dataset,
        "v05_fno.pt": args.fno_checkpoint,
        "v07_diffusion.pt": args.ddpm_checkpoint,
        "v07_diffusion_candidates.npz": args.ddpm_candidates,
        "v08_refined_candidates.npz": args.v08_refined,
    }

    manifest = {
        "stage": "V09",
        "git_head_before_v09": args.git_head,
        "tracked_result_sha256": {
            name: sha256_file(path)
            for name, path in tracked_results.items()
        },
        "local_ignored_artifact_sha256": {
            name: sha256_file(path)
            for name, path in local_artifacts.items()
        },
        "local_ignored_artifact_bytes": {
            name: path.stat().st_size
            for name, path in local_artifacts.items()
        },
        "reproduction_contract": (
            "Tracked code/results plus the recorded SHA256 values identify "
            "the local dataset/checkpoints/candidate arrays used by V09. "
            "The final CLOSE step may archive the project once."
        ),
    }

    args.result.parent.mkdir(parents=True, exist_ok=True)

    args.result.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    args.manifest.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_csv(args.csv, rows)

    print("V09 cross-project summary finished")
    print(
        "project1 compliance reduction="
        f"{percentage(simp_reduction):.2f}%"
    )
    print(
        "project2 FNO speedup/order/residual="
        f"{fno_speedup:.2f}x/"
        f"{percentage(pairwise):.1f}%/"
        f"{fno_residual:.3e}"
    )
    print(
        "project3 refined/uniform30="
        f"{refined_over_uniform:.5f}; "
        "better conditions="
        f"{diffusion_better_count}/{condition_count}"
    )
    print(
        "project3 connected raw/refined selected="
        f"{screening_connected}/{screening_total}, "
        f"{selected_connected_after}/{selected_count}"
    )


if __name__ == "__main__":
    main()
