"""V02: воспроизводимый FEM-baseline для консоли и half-MBB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    sparse_symmetry_error,
)


def case_payload(mesh, stiffness, condition):
    result = solve_elasticity(stiffness, condition)
    fixed = np.unique(condition.fixed_dofs)
    return {
        "name": condition.name,
        "volume_fraction_reserved_for_v03": condition.volume_fraction,
        "fixed_dofs": int(fixed.size),
        "free_dofs": int(result.free_dofs.size),
        "compliance": result.compliance,
        "strain_energy": result.strain_energy,
        "relative_residual": result.relative_residual,
        "energy_identity_error": result.energy_identity_error,
        "force_balance_error": result.force_balance_error,
        "max_abs_displacement": float(np.max(np.abs(result.displacement))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Куда записать детерминированный JSON с результатами V02.",
    )
    args = parser.parse_args()

    mesh = StructuredQuadMesh(
        nelx=96,
        nely=32,
        length=3.0,
        height=1.0,
    )
    young = 1.0
    poisson = 0.3
    thickness = 1.0

    stiffness = assemble_stiffness(
        mesh,
        young=young,
        poisson=poisson,
        thickness=thickness,
    )
    symmetry_error = sparse_symmetry_error(stiffness)

    cantilever = cantilever_condition(
        mesh,
        volume_fraction=0.4,
        load_y_fraction=0.5,
        load_angle_deg=-90.0,
        load_magnitude=1.0,
    )
    mbb = half_mbb_condition(
        mesh,
        volume_fraction=0.4,
        load_magnitude=1.0,
    )

    payload = {
        "stage": "V02",
        "model": "2D linear elasticity, plane stress, Q4, 2x2 Gauss",
        "mesh": {
            "nelx": mesh.nelx,
            "nely": mesh.nely,
            "length": mesh.length,
            "height": mesh.height,
            "hx": mesh.hx,
            "hy": mesh.hy,
            "elements": mesh.nelems,
            "nodes": mesh.nnodes,
            "dofs": mesh.ndof,
        },
        "material": {
            "young": young,
            "poisson": poisson,
            "thickness": thickness,
        },
        "matrix": {
            "nnz": int(stiffness.nnz),
            "symmetry_error": symmetry_error,
        },
        "cases": {
            "cantilever": case_payload(mesh, stiffness, cantilever),
            "half_mbb": case_payload(mesh, stiffness, mbb),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("V02 FEM baseline")
    print(
        f"mesh={mesh.nelx}x{mesh.nely}, "
        f"elements={mesh.nelems}, nodes={mesh.nnodes}, dofs={mesh.ndof}"
    )
    print(f"K nnz={stiffness.nnz}, symmetry_error={symmetry_error:.3e}")
    for name, case in payload["cases"].items():
        print(
            f"{name}: J={case['compliance']:.12g}, "
            f"residual={case['relative_residual']:.3e}, "
            f"energy_error={case['energy_identity_error']:.3e}, "
            f"balance_error={case['force_balance_error']:.3e}"
        )


if __name__ == "__main__":
    main()
