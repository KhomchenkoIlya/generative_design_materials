"""Минимальный 2D FEM-решатель линейной упругости для главы 10.

Дискретизация:
    * прямоугольная структурированная сетка Q4;
    * плоское напряжённое состояние;
    * 2x2 квадратура Гаусса;
    * нулевые условия Дирихле задаются списком степеней свободы.

Модуль намеренно не содержит логики SIMP. V02 проверяет физический решатель
на сплошном материале. В V03 в assemble_stiffness будут передаваться
поэлементные множители жёсткости.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class StructuredQuadMesh:
    """Прямоугольная Q4-сетка с двумя степенями свободы в каждом узле."""

    nelx: int
    nely: int
    length: float = 3.0
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.nelx <= 0 or self.nely <= 0:
            raise ValueError("nelx и nely должны быть положительными.")
        if self.length <= 0.0 or self.height <= 0.0:
            raise ValueError("Размеры области должны быть положительными.")

    @property
    def nnodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1)

    @property
    def ndof(self) -> int:
        return 2 * self.nnodes

    @property
    def nelems(self) -> int:
        return self.nelx * self.nely

    @property
    def hx(self) -> float:
        return self.length / self.nelx

    @property
    def hy(self) -> float:
        return self.height / self.nely

    def node_id(self, ix: int, iy: int) -> int:
        if not (0 <= ix <= self.nelx and 0 <= iy <= self.nely):
            raise IndexError("Индекс узла вне сетки.")
        return ix * (self.nely + 1) + iy

    def coordinates(self) -> np.ndarray:
        """Координаты узлов, совместимые с node_id."""

        xs = np.linspace(0.0, self.length, self.nelx + 1)
        ys = np.linspace(0.0, self.height, self.nely + 1)
        coords = np.empty((self.nnodes, 2), dtype=float)

        k = 0
        for x in xs:
            for y in ys:
                coords[k] = (x, y)
                k += 1
        return coords

    def elements(self) -> np.ndarray:
        """Узлы элементов в положительной, против часовой стрелки, ориентации."""

        elems = np.empty((self.nelems, 4), dtype=np.int64)
        k = 0
        for ix in range(self.nelx):
            for iy in range(self.nely):
                elems[k] = (
                    self.node_id(ix, iy),
                    self.node_id(ix + 1, iy),
                    self.node_id(ix + 1, iy + 1),
                    self.node_id(ix, iy + 1),
                )
                k += 1
        return elems

    def edof_matrix(self) -> np.ndarray:
        """Глобальные степени свободы всех Q4-элементов."""

        elems = self.elements()
        edof = np.empty((self.nelems, 8), dtype=np.int64)
        edof[:, 0::2] = 2 * elems
        edof[:, 1::2] = 2 * elems + 1
        return edof

    def left_nodes(self) -> np.ndarray:
        return np.array(
            [self.node_id(0, iy) for iy in range(self.nely + 1)],
            dtype=np.int64,
        )

    def dofs_for_nodes(self, nodes: np.ndarray) -> np.ndarray:
        nodes = np.asarray(nodes, dtype=np.int64)
        dofs = np.empty(2 * nodes.size, dtype=np.int64)
        dofs[0::2] = 2 * nodes
        dofs[1::2] = 2 * nodes + 1
        return dofs


@dataclass(frozen=True)
class ProblemCondition:
    """Данные задачи, не выбираемые оптимизатором."""

    name: str
    fixed_dofs: np.ndarray
    force: np.ndarray
    volume_fraction: float


@dataclass(frozen=True)
class FEMResult:
    displacement: np.ndarray
    reactions: np.ndarray
    free_dofs: np.ndarray
    compliance: float
    strain_energy: float
    relative_residual: float
    energy_identity_error: float
    force_balance_error: float


def plane_stress_matrix(young: float = 1.0, poisson: float = 0.3) -> np.ndarray:
    """Матрица закона Гука для плоского напряжённого состояния."""

    if young <= 0.0:
        raise ValueError("Модуль Юнга должен быть положительным.")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("Некорректный коэффициент Пуассона.")

    factor = young / (1.0 - poisson**2)
    return factor * np.array(
        [
            [1.0, poisson, 0.0],
            [poisson, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poisson)],
        ],
        dtype=float,
    )


def quad4_stiffness(
    element_coordinates: np.ndarray,
    young: float = 1.0,
    poisson: float = 0.3,
    thickness: float = 1.0,
) -> np.ndarray:
    """8x8 матрица жёсткости билинейного четырёхузлового элемента."""

    coords = np.asarray(element_coordinates, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("Для Q4 нужны координаты формы (4, 2).")
    if thickness <= 0.0:
        raise ValueError("Толщина должна быть положительной.")

    constitutive = plane_stress_matrix(young, poisson)
    ke = np.zeros((8, 8), dtype=float)
    gauss = 1.0 / math.sqrt(3.0)

    for xi in (-gauss, gauss):
        for eta in (-gauss, gauss):
            dshape_dnatural = 0.25 * np.array(
                [
                    [-(1.0 - eta), -(1.0 - xi)],
                    [+(1.0 - eta), -(1.0 + xi)],
                    [+(1.0 + eta), +(1.0 + xi)],
                    [-(1.0 + eta), +(1.0 - xi)],
                ],
                dtype=float,
            )

            jacobian = dshape_dnatural.T @ coords
            det_jacobian = float(np.linalg.det(jacobian))
            if det_jacobian <= 0.0:
                raise ValueError("Элемент имеет неположительный якобиан.")

            dshape_dx = dshape_dnatural @ np.linalg.inv(jacobian)

            bmat = np.zeros((3, 8), dtype=float)
            for node in range(4):
                dndx, dndy = dshape_dx[node]
                bmat[0, 2 * node] = dndx
                bmat[1, 2 * node + 1] = dndy
                bmat[2, 2 * node] = dndy
                bmat[2, 2 * node + 1] = dndx

            ke += (
                bmat.T
                @ constitutive
                @ bmat
                * det_jacobian
                * thickness
            )

    return 0.5 * (ke + ke.T)


def reference_element_stiffness(
    mesh: StructuredQuadMesh,
    young: float = 1.0,
    poisson: float = 0.3,
    thickness: float = 1.0,
) -> np.ndarray:
    """Матрица первого элемента; на равномерной сетке она общая для всех элементов."""

    coords = mesh.coordinates()
    first_nodes = mesh.elements()[0]
    return quad4_stiffness(
        coords[first_nodes],
        young=young,
        poisson=poisson,
        thickness=thickness,
    )


def assemble_stiffness(
    mesh: StructuredQuadMesh,
    *,
    young: float = 1.0,
    poisson: float = 0.3,
    thickness: float = 1.0,
    element_factors: np.ndarray | None = None,
) -> csr_matrix:
    """Собрать глобальную матрицу жёсткости.

    element_factors — безразмерные множители поэлементной жёсткости.
    В V02 они равны единице; V03 будет передавать сюда SIMP-интерполяцию.
    """

    if element_factors is None:
        factors = np.ones(mesh.nelems, dtype=float)
    else:
        factors = np.asarray(element_factors, dtype=float)
        if factors.shape != (mesh.nelems,):
            raise ValueError("element_factors должен иметь длину nelems.")
        if np.any(factors <= 0.0):
            raise ValueError("Все множители жёсткости должны быть положительными.")

    ke = reference_element_stiffness(
        mesh,
        young=young,
        poisson=poisson,
        thickness=thickness,
    )
    edof = mesh.edof_matrix()

    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    values = (factors[:, None] * ke.reshape(1, 64)).ravel()

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(mesh.ndof, mesh.ndof),
    ).tocsr()

    # Теоретически K симметрична. Усреднение удаляет только машинный перекос.
    return (0.5 * (matrix + matrix.T)).tocsr()


def cantilever_condition(
    mesh: StructuredQuadMesh,
    *,
    volume_fraction: float = 0.4,
    load_y_fraction: float = 0.5,
    load_angle_deg: float = -90.0,
    load_magnitude: float = 1.0,
) -> ProblemCondition:
    """Консоль: левая грань закреплена, нагрузка задаётся на правой границе."""

    if not (0.0 < volume_fraction <= 1.0):
        raise ValueError("volume_fraction должна лежать в (0, 1].")
    if not (0.0 <= load_y_fraction <= 1.0):
        raise ValueError("load_y_fraction должна лежать в [0, 1].")
    if load_magnitude <= 0.0:
        raise ValueError("load_magnitude должна быть положительной.")

    fixed = np.unique(mesh.dofs_for_nodes(mesh.left_nodes()))

    iy = int(round(load_y_fraction * mesh.nely))
    load_node = mesh.node_id(mesh.nelx, iy)
    angle = math.radians(load_angle_deg)

    force = np.zeros(mesh.ndof, dtype=float)
    force[2 * load_node] = load_magnitude * math.cos(angle)
    force[2 * load_node + 1] = load_magnitude * math.sin(angle)

    return ProblemCondition(
        name="cantilever",
        fixed_dofs=fixed,
        force=force,
        volume_fraction=volume_fraction,
    )


def half_mbb_condition(
    mesh: StructuredQuadMesh,
    *,
    volume_fraction: float = 0.4,
    load_magnitude: float = 1.0,
) -> ProblemCondition:
    """Half-MBB: ux=0 слева, uy=0 в правом нижнем узле, сила вниз слева сверху."""

    if not (0.0 < volume_fraction <= 1.0):
        raise ValueError("volume_fraction должна лежать в (0, 1].")
    if load_magnitude <= 0.0:
        raise ValueError("load_magnitude должна быть положительной.")

    left = mesh.left_nodes()
    fixed_x = 2 * left
    lower_right = mesh.node_id(mesh.nelx, 0)
    fixed = np.unique(
        np.concatenate(
            [fixed_x, np.array([2 * lower_right + 1], dtype=np.int64)]
        )
    )

    upper_left = mesh.node_id(0, mesh.nely)
    force = np.zeros(mesh.ndof, dtype=float)
    force[2 * upper_left + 1] = -load_magnitude

    return ProblemCondition(
        name="half_mbb",
        fixed_dofs=fixed,
        force=force,
        volume_fraction=volume_fraction,
    )


def solve_elasticity(
    stiffness: csr_matrix,
    condition: ProblemCondition,
) -> FEMResult:
    """Решить K_ff U_f = F_f при нулевых заданных перемещениях."""

    if stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("Матрица жёсткости должна быть квадратной.")
    if stiffness.shape[0] != condition.force.size:
        raise ValueError("Размер K не согласован с F.")

    ndof = stiffness.shape[0]
    fixed = np.unique(np.asarray(condition.fixed_dofs, dtype=np.int64))
    if fixed.size == 0:
        raise ValueError("Не заданы условия Дирихле.")
    if fixed[0] < 0 or fixed[-1] >= ndof:
        raise ValueError("fixed_dofs содержит индекс вне матрицы.")

    free_mask = np.ones(ndof, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)

    kff = stiffness[free][:, free].tocsr()
    ff = condition.force[free]
    uf = spsolve(kff, ff)

    if not np.all(np.isfinite(uf)):
        raise RuntimeError("Линейная система дала нечисловое решение.")

    displacement = np.zeros(ndof, dtype=float)
    displacement[free] = uf

    free_residual = kff @ uf - ff
    denominator = max(float(np.linalg.norm(ff)), np.finfo(float).eps)
    relative_residual = float(np.linalg.norm(free_residual) / denominator)

    reactions = np.asarray(stiffness @ displacement - condition.force)
    compliance = float(condition.force @ displacement)
    strain_energy = float(0.5 * displacement @ (stiffness @ displacement))

    energy_denominator = max(abs(compliance), np.finfo(float).eps)
    energy_identity_error = abs(compliance - 2.0 * strain_energy) / energy_denominator

    reaction_resultant = np.array(
        [
            reactions[0::2].sum(),
            reactions[1::2].sum(),
        ],
        dtype=float,
    )
    external_resultant = np.array(
        [
            condition.force[0::2].sum(),
            condition.force[1::2].sum(),
        ],
        dtype=float,
    )
    force_balance_error = float(
        np.linalg.norm(reaction_resultant + external_resultant)
        / max(np.linalg.norm(external_resultant), np.finfo(float).eps)
    )

    return FEMResult(
        displacement=displacement,
        reactions=reactions,
        free_dofs=free,
        compliance=compliance,
        strain_energy=strain_energy,
        relative_residual=relative_residual,
        energy_identity_error=float(energy_identity_error),
        force_balance_error=force_balance_error,
    )


def sparse_symmetry_error(matrix: csr_matrix) -> float:
    """Максимальный абсолютный элемент K-K^T."""

    diff = matrix - matrix.T
    if diff.nnz == 0:
        return 0.0
    return float(np.max(np.abs(diff.data)))
