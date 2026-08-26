"""V03: два SIMP-запуска, gradient check и проверка повторным FEM на других сетках."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

CHAPTER_DIR = Path(__file__).resolve().parents[1]
if str(CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(CHAPTER_DIR))

from common.fem2d import (  # noqa: E402
    StructuredQuadMesh,
    assemble_stiffness,
    cantilever_condition,
    half_mbb_condition,
    solve_elasticity,
)
from project1_topopt.topopt import (  # noqa: E402
    SIMPConfig,
    finite_difference_gradient_check,
    optimize_simp,
    resample_factor_two,
    write_density_png,
    write_history_csv,
)


def evaluate_physical_density(
    density: np.ndarray,
    mesh: StructuredQuadMesh,
    *,
    case: str,
    config: SIMPConfig,
) -> dict[str, float]:
    if case != "cantilever":
        raise ValueError("V03 mesh-check реализован для основной консоли.")

    condition = cantilever_condition(
        mesh,
        volume_fraction=0.4,
        load_y_fraction=0.5,
        load_angle_deg=-90.0,
        load_magnitude=1.0,
    )

    factors = (
        config.emin_ratio
        + np.asarray(density) ** config.penal
        * (1.0 - config.emin_ratio)
    )

    stiffness = assemble_stiffness(
        mesh,
        young=1.0,
        poisson=0.3,
        thickness=1.0,
        element_factors=factors,
    )
    fem = solve_elasticity(stiffness, condition)

    return {
        "compliance": fem.compliance,
        "volume_fraction": float(np.mean(density)),
        "relative_residual": fem.relative_residual,
    }


def result_payload(result, runtime_seconds: float) -> dict[str, object]:
    return {
        "initial_compliance": result.initial_compliance,
        "final_compliance": result.compliance,
        "relative_improvement": (
            result.initial_compliance - result.compliance
        ) / result.initial_compliance,
        "volume_fraction": result.volume_fraction,
        "iterations": result.iterations,
        "converged": result.converged,
        "final_change": result.final_change,
        "relative_residual": result.relative_residual,
        "energy_identity_error": result.energy_identity_error,
        "force_balance_error": result.force_balance_error,
        "density_min": float(np.min(result.physical_density)),
        "density_max": float(np.max(result.physical_density)),
        "density_gray_fraction": float(
            np.mean(
                (result.physical_density > 0.1)
                & (result.physical_density < 0.9)
            )
        ),
        "runtime_seconds": runtime_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cant-history", type=Path, required=True)
    parser.add_argument("--mbb-history", type=Path, required=True)
    parser.add_argument("--cant-png", type=Path, required=True)
    parser.add_argument("--mbb-png", type=Path, required=True)
    args = parser.parse_args()

    config = SIMPConfig(
        penal=3.0,
        rmin=1.5,
        emin_ratio=1.0e-9,
        move=0.2,
        tolerance=1.0e-2,
        max_iterations=250,
    )

    small_mesh = StructuredQuadMesh(
        nelx=12,
        nely=4,
        length=3.0,
        height=1.0,
    )
    small_condition = cantilever_condition(
        small_mesh,
        volume_fraction=0.4,
        load_y_fraction=0.5,
        load_angle_deg=-90.0,
        load_magnitude=1.0,
    )
    gradient_check = finite_difference_gradient_check(
        small_mesh,
        small_condition,
        config,
        seed=20260813,
        count=6,
        step=1.0e-6,
    )

    mesh = StructuredQuadMesh(
        nelx=96,
        nely=32,
        length=3.0,
        height=1.0,
    )

    cantilever = cantilever_condition(
        mesh,
        volume_fraction=0.4,
        load_y_fraction=0.5,
        load_angle_deg=-90.0,
        load_magnitude=1.0,
    )
    half_mbb = half_mbb_condition(
        mesh,
        volume_fraction=0.4,
        load_magnitude=1.0,
    )

    start = time.perf_counter()
    cant_result = optimize_simp(mesh, cantilever, config)
    cant_runtime = time.perf_counter() - start

    start = time.perf_counter()
    mbb_result = optimize_simp(mesh, half_mbb, config)
    mbb_runtime = time.perf_counter() - start

    write_history_csv(cant_result.history, args.cant_history)
    write_history_csv(mbb_result.history, args.mbb_history)

    write_density_png(
        cant_result.physical_density,
        mesh.nelx,
        mesh.nely,
        args.cant_png,
    )
    write_density_png(
        mbb_result.physical_density,
        mesh.nelx,
        mesh.nely,
        args.mbb_png,
    )

    coarse_density, coarse_nelx, coarse_nely = resample_factor_two(
        cant_result.physical_density,
        mesh.nelx,
        mesh.nely,
        mode="coarsen",
    )
    fine_density, fine_nelx, fine_nely = resample_factor_two(
        cant_result.physical_density,
        mesh.nelx,
        mesh.nely,
        mode="refine",
    )

    coarse_mesh = StructuredQuadMesh(
        nelx=coarse_nelx,
        nely=coarse_nely,
        length=3.0,
        height=1.0,
    )
    fine_mesh = StructuredQuadMesh(
        nelx=fine_nelx,
        nely=fine_nely,
        length=3.0,
        height=1.0,
    )

    coarse_eval = evaluate_physical_density(
        coarse_density,
        coarse_mesh,
        case="cantilever",
        config=config,
    )
    reference_eval = evaluate_physical_density(
        cant_result.physical_density,
        mesh,
        case="cantilever",
        config=config,
    )
    fine_eval = evaluate_physical_density(
        fine_density,
        fine_mesh,
        case="cantilever",
        config=config,
    )

    reference_j = reference_eval["compliance"]

    mesh_check = {
        "interpretation": (
            "Same optimized physical density is re-evaluated after exact "
            "factor-two averaging/prolongation; this checks FEM/discretization "
            "sensitivity, not convergence of separately re-optimized designs."
        ),
        "48x16": {
            **coarse_eval,
            "relative_to_96x32": (
                coarse_eval["compliance"] - reference_j
            ) / reference_j,
        },
        "96x32": {
            **reference_eval,
            "relative_to_96x32": 0.0,
        },
        "192x64": {
            **fine_eval,
            "relative_to_96x32": (
                fine_eval["compliance"] - reference_j
            ) / reference_j,
        },
    }

    payload = {
        "stage": "V03",
        "config": {
            "penal": config.penal,
            "rmin_elements": config.rmin,
            "emin_ratio": config.emin_ratio,
            "move": config.move,
            "tolerance": config.tolerance,
            "max_iterations": config.max_iterations,
            "volume_fraction": 0.4,
        },
        "gradient_check": gradient_check,
        "cases": {
            "cantilever": result_payload(cant_result, cant_runtime),
            "half_mbb": result_payload(mbb_result, mbb_runtime),
        },
        "cantilever_mesh_reevaluation": mesh_check,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("V03 SIMP + density filter + OC")
    print(
        "gradient max relative error="
        f"{gradient_check['max_relative_error']:.3e}"
    )
    for name in ("cantilever", "half_mbb"):
        case = payload["cases"][name]
        print(
            f"{name}: J0={case['initial_compliance']:.6f}, "
            f"J={case['final_compliance']:.6f}, "
            f"V={case['volume_fraction']:.9f}, "
            f"iters={case['iterations']}, "
            f"change={case['final_change']:.3e}, "
            f"residual={case['relative_residual']:.3e}"
        )
    print(
        "cantilever re-evaluation: "
        f"48x16={mesh_check['48x16']['compliance']:.6f}, "
        f"96x32={mesh_check['96x32']['compliance']:.6f}, "
        f"192x64={mesh_check['192x64']['compliance']:.6f}"
    )


if __name__ == "__main__":
    main()
