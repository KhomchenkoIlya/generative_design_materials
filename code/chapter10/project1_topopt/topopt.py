"""SIMP topology optimization поверх проверенного FEM-ядра V02.

Проектная переменная x фильтруется линейным density filter:
    rho = H x / Hs.

Физическая жёсткость элемента:
    E_e / E0 = Emin + rho_e**p * (1 - Emin).

Производная compliance переносится на x точным правилом цепочки через H.
Ограничение объёма также вычисляется по физической плотности rho.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
import struct
import zlib

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from common.fem2d import (
    ProblemCondition,
    StructuredQuadMesh,
    assemble_stiffness,
    reference_element_stiffness,
    solve_elasticity,
)


@dataclass(frozen=True)
class SIMPConfig:
    penal: float = 3.0
    rmin: float = 1.5
    emin_ratio: float = 1.0e-9
    move: float = 0.2
    tolerance: float = 1.0e-2
    max_iterations: int = 250

    def __post_init__(self) -> None:
        if self.penal <= 1.0:
            raise ValueError("penal должен быть больше 1.")
        if self.rmin <= 0.0:
            raise ValueError("rmin должен быть положительным.")
        if not (0.0 < self.emin_ratio < 1.0):
            raise ValueError("emin_ratio должен лежать в (0, 1).")
        if not (0.0 < self.move <= 1.0):
            raise ValueError("move должен лежать в (0, 1].")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance должна быть положительной.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations должен быть положительным.")


@dataclass(frozen=True)
class DensityFilter:
    matrix: csr_matrix
    row_sum: np.ndarray

    def apply(self, design: np.ndarray) -> np.ndarray:
        design = np.asarray(design, dtype=float)
        return np.asarray(self.matrix @ design).ravel() / self.row_sum

    def pullback(self, physical_gradient: np.ndarray) -> np.ndarray:
        physical_gradient = np.asarray(physical_gradient, dtype=float)
        return np.asarray(
            self.matrix.T @ (physical_gradient / self.row_sum)
        ).ravel()


@dataclass(frozen=True)
class DesignAnalysis:
    physical_density: np.ndarray
    displacement: np.ndarray
    compliance: float
    volume_fraction: float
    relative_residual: float
    energy_identity_error: float
    force_balance_error: float
    gradient: np.ndarray
    volume_gradient: np.ndarray


@dataclass(frozen=True)
class OptimizationResult:
    design: np.ndarray
    physical_density: np.ndarray
    compliance: float
    initial_compliance: float
    volume_fraction: float
    iterations: int
    converged: bool
    final_change: float
    relative_residual: float
    energy_identity_error: float
    force_balance_error: float
    history: tuple[dict[str, float], ...]


def build_density_filter(
    mesh: StructuredQuadMesh,
    rmin: float,
) -> DensityFilter:
    """Классический линейный фильтр в координатах элементов."""

    if rmin <= 0.0:
        raise ValueError("rmin должен быть положительным.")

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    reach = int(math.ceil(rmin))

    for ix in range(mesh.nelx):
        for iy in range(mesh.nely):
            row = ix * mesh.nely + iy

            jx_min = max(0, ix - reach + 1)
            jx_max = min(mesh.nelx, ix + reach)
            jy_min = max(0, iy - reach + 1)
            jy_max = min(mesh.nely, iy + reach)

            for jx in range(jx_min, jx_max):
                for jy in range(jy_min, jy_max):
                    distance = math.hypot(ix - jx, iy - jy)
                    weight = max(0.0, rmin - distance)
                    if weight == 0.0:
                        continue

                    col = jx * mesh.nely + jy
                    rows.append(row)
                    cols.append(col)
                    values.append(weight)

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(mesh.nelems, mesh.nelems),
    ).tocsr()

    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(row_sum <= 0.0):
        raise RuntimeError("Фильтр содержит строку с нулевой суммой весов.")

    return DensityFilter(matrix=matrix, row_sum=row_sum)


def analyze_design(
    mesh: StructuredQuadMesh,
    condition: ProblemCondition,
    design: np.ndarray,
    density_filter: DensityFilter,
    config: SIMPConfig,
) -> DesignAnalysis:
    """Точный FEM-анализ и производные compliance/volume по design x."""

    design = np.asarray(design, dtype=float)
    if design.shape != (mesh.nelems,):
        raise ValueError("design должен иметь длину mesh.nelems.")
    if np.any(design < 0.0) or np.any(design > 1.0):
        raise ValueError("design должен лежать в [0, 1].")

    rho = density_filter.apply(design)

    factors = (
        config.emin_ratio
        + rho**config.penal * (1.0 - config.emin_ratio)
    )

    stiffness = assemble_stiffness(
        mesh,
        young=1.0,
        poisson=0.3,
        thickness=1.0,
        element_factors=factors,
    )
    fem = solve_elasticity(stiffness, condition)

    edof = mesh.edof_matrix()
    ke0 = reference_element_stiffness(
        mesh,
        young=1.0,
        poisson=0.3,
        thickness=1.0,
    )
    element_u = fem.displacement[edof]
    element_energy = np.einsum(
        "ei,ij,ej->e",
        element_u,
        ke0,
        element_u,
    )

    dcompliance_drho = (
        -config.penal
        * (1.0 - config.emin_ratio)
        * rho ** (config.penal - 1.0)
        * element_energy
    )
    gradient = density_filter.pullback(dcompliance_drho)

    # V = mean(rho). Поскольку rho = Hx/Hs, цепочка та же.
    dvolume_drho = np.full(mesh.nelems, 1.0 / mesh.nelems)
    volume_gradient = density_filter.pullback(dvolume_drho)

    return DesignAnalysis(
        physical_density=rho,
        displacement=fem.displacement,
        compliance=fem.compliance,
        volume_fraction=float(np.mean(rho)),
        relative_residual=fem.relative_residual,
        energy_identity_error=fem.energy_identity_error,
        force_balance_error=fem.force_balance_error,
        gradient=gradient,
        volume_gradient=volume_gradient,
    )


def oc_update(
    design: np.ndarray,
    compliance_gradient: np.ndarray,
    volume_gradient: np.ndarray,
    density_filter: DensityFilter,
    volume_fraction: float,
    move: float,
) -> np.ndarray:
    """Один OC-шаг с бисекцией множителя ограничения объёма."""

    design = np.asarray(design, dtype=float)
    compliance_gradient = np.asarray(compliance_gradient, dtype=float)
    volume_gradient = np.asarray(volume_gradient, dtype=float)

    if not (0.0 < volume_fraction < 1.0):
        raise ValueError("volume_fraction должна лежать в (0, 1).")

    lower = 0.0
    upper = 1.0e9
    candidate = design.copy()

    for _ in range(100):
        lagrange = 0.5 * (lower + upper)

        ratio = np.maximum(
            1.0e-30,
            -compliance_gradient / (volume_gradient * lagrange),
        )

        candidate = design * np.sqrt(ratio)
        candidate = np.maximum(design - move, candidate)
        candidate = np.minimum(design + move, candidate)
        candidate = np.clip(candidate, 0.0, 1.0)

        physical = density_filter.apply(candidate)

        if float(np.mean(physical)) > volume_fraction:
            lower = lagrange
        else:
            upper = lagrange

        if (upper - lower) / (upper + lower + 1.0e-30) < 1.0e-8:
            break

    return candidate


def optimize_simp(
    mesh: StructuredQuadMesh,
    condition: ProblemCondition,
    config: SIMPConfig,
) -> OptimizationResult:
    """Полный SIMP + density filter + OC."""

    density_filter = build_density_filter(mesh, config.rmin)
    design = np.full(mesh.nelems, condition.volume_fraction, dtype=float)

    initial = analyze_design(
        mesh,
        condition,
        design,
        density_filter,
        config,
    )

    history: list[dict[str, float]] = []
    converged = False
    final_change = float("inf")

    for iteration in range(1, config.max_iterations + 1):
        analysis = analyze_design(
            mesh,
            condition,
            design,
            density_filter,
            config,
        )

        updated = oc_update(
            design,
            analysis.gradient,
            analysis.volume_gradient,
            density_filter,
            condition.volume_fraction,
            config.move,
        )

        final_change = float(np.max(np.abs(updated - design)))

        history.append(
            {
                "iteration": float(iteration),
                "compliance": analysis.compliance,
                "volume_fraction": analysis.volume_fraction,
                "max_change": final_change,
            }
        )

        design = updated

        if final_change < config.tolerance:
            converged = True
            break

    final = analyze_design(
        mesh,
        condition,
        design,
        density_filter,
        config,
    )

    return OptimizationResult(
        design=design,
        physical_density=final.physical_density,
        compliance=final.compliance,
        initial_compliance=initial.compliance,
        volume_fraction=final.volume_fraction,
        iterations=len(history),
        converged=converged,
        final_change=final_change,
        relative_residual=final.relative_residual,
        energy_identity_error=final.energy_identity_error,
        force_balance_error=final.force_balance_error,
        history=tuple(history),
    )


def finite_difference_gradient_check(
    mesh: StructuredQuadMesh,
    condition: ProblemCondition,
    config: SIMPConfig,
    *,
    seed: int = 20260813,
    count: int = 6,
    step: float = 1.0e-6,
) -> dict[str, object]:
    """Сравнить аналитическую производную compliance с центральной разностью."""

    rng = np.random.default_rng(seed)
    design = 0.25 + 0.50 * rng.random(mesh.nelems)
    density_filter = build_density_filter(mesh, config.rmin)

    analysis = analyze_design(
        mesh,
        condition,
        design,
        density_filter,
        config,
    )

    indices = np.linspace(
        0,
        mesh.nelems - 1,
        num=min(count, mesh.nelems),
        dtype=int,
    )

    rows: list[dict[str, float]] = []

    for index in indices:
        plus = design.copy()
        minus = design.copy()
        plus[index] += step
        minus[index] -= step

        j_plus = analyze_design(
            mesh,
            condition,
            plus,
            density_filter,
            config,
        ).compliance
        j_minus = analyze_design(
            mesh,
            condition,
            minus,
            density_filter,
            config,
        ).compliance

        finite_difference = (j_plus - j_minus) / (2.0 * step)
        analytic = float(analysis.gradient[index])

        relative_error = abs(analytic - finite_difference) / max(
            1.0,
            abs(analytic),
            abs(finite_difference),
        )

        rows.append(
            {
                "index": float(index),
                "analytic": analytic,
                "finite_difference": finite_difference,
                "relative_error": relative_error,
            }
        )

    return {
        "seed": seed,
        "step": step,
        "max_relative_error": max(row["relative_error"] for row in rows),
        "checks": rows,
    }


def write_history_csv(
    history: tuple[dict[str, float], ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "iteration",
                "compliance",
                "volume_fraction",
                "max_change",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def write_density_png(
    density: np.ndarray,
    nelx: int,
    nely: int,
    path: Path,
    *,
    scale: int = 6,
) -> None:
    """Записать grayscale PNG без matplotlib: чёрное = материал."""

    density = np.asarray(density, dtype=float)
    if density.shape != (nelx * nely,):
        raise ValueError("Некорректный размер density.")

    field = density.reshape(nelx, nely).T
    field = np.flipud(field)
    gray = np.rint(255.0 * (1.0 - np.clip(field, 0.0, 1.0))).astype(np.uint8)

    image = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)
    height, width = image.shape

    raw = b"".join(
        b"\x00" + image[row].tobytes()
        for row in range(height)
    )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def resample_factor_two(
    density: np.ndarray,
    nelx: int,
    nely: int,
    *,
    mode: str,
) -> tuple[np.ndarray, int, int]:
    """Только точное удвоение/усреднение в 2 раза для mesh-check V03."""

    field = np.asarray(density, dtype=float).reshape(nelx, nely)

    if mode == "coarsen":
        if nelx % 2 or nely % 2:
            raise ValueError("Для coarsen размеры должны быть чётными.")
        coarse = field.reshape(
            nelx // 2,
            2,
            nely // 2,
            2,
        ).mean(axis=(1, 3))
        return coarse.ravel(), nelx // 2, nely // 2

    if mode == "refine":
        fine = np.repeat(np.repeat(field, 2, axis=0), 2, axis=1)
        return fine.ravel(), 2 * nelx, 2 * nely

    raise ValueError("mode должен быть 'coarsen' или 'refine'.")
