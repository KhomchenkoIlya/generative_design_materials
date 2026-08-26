"""Компактный 2D Fourier Neural Operator для FEM-суррогата V05."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


INPUT_CHANNELS = 9
OUTPUT_CHANNELS = 2


def coordinate_grid(
    height: int,
    width: int,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Два канала x/L и y/H на nodal grid."""

    x = np.linspace(0.0, 1.0, width, dtype=dtype)
    y = np.linspace(0.0, 1.0, height, dtype=dtype)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.stack((xx, yy), axis=0).astype(dtype)


def build_raw_inputs(
    rho_node: np.ndarray,
    force: np.ndarray,
    condition: np.ndarray,
) -> np.ndarray:
    """Собрать 9 spatial-каналов.

    Каналы:
        rho,
        Fx, Fy,
        x/L, y/H,
        f_vol, y_F/H, cos(theta_F), sin(theta_F).
    """

    rho_node = np.asarray(rho_node, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32)
    condition = np.asarray(condition, dtype=np.float32)

    if rho_node.ndim != 3:
        raise ValueError("rho_node должен иметь shape (N,H,W).")
    if force.ndim != 4 or force.shape[1] != 2:
        raise ValueError("force должен иметь shape (N,2,H,W).")
    if condition.ndim != 2 or condition.shape[1] != 4:
        raise ValueError("condition должен иметь shape (N,4).")

    n, height, width = rho_node.shape
    if force.shape != (n, 2, height, width):
        raise ValueError("rho_node и force имеют несовместимые shape.")
    if condition.shape[0] != n:
        raise ValueError("condition и spatial arrays имеют разный N.")

    coords = coordinate_grid(height, width)
    coords = np.broadcast_to(
        coords[None, ...],
        (n, 2, height, width),
    )

    global_channels = np.broadcast_to(
        condition[:, :, None, None],
        (n, 4, height, width),
    )

    return np.concatenate(
        (
            rho_node[:, None, :, :],
            force,
            coords,
            global_channels,
        ),
        axis=1,
    ).astype(np.float32)


def channel_statistics(
    raw_inputs: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std только по train samples и spatial axes."""

    train = np.asarray(raw_inputs[train_mask], dtype=np.float64)
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(std, 1.0e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def output_rms(
    displacement: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """RMS каждого компонента U на train; без вычитания среднего."""

    train = np.asarray(displacement[train_mask], dtype=np.float64)
    rms = np.sqrt(np.mean(train**2, axis=(0, 2, 3), keepdims=True))
    rms = np.maximum(rms, 1.0e-8)
    return rms.astype(np.float32)


class SpectralConv2d(nn.Module):
    """Низкочастотный Fourier integral operator на регулярной сетке."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_y: int,
        modes_x: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x

        scale = 1.0 / math.sqrt(in_channels * out_channels)

        self.weight_top = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_y,
                modes_x,
                dtype=torch.cfloat,
            )
        )
        self.weight_bottom = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes_y,
                modes_x,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def _multiply(
        spectrum: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bixy,ioxy->boxy",
            spectrum,
            weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape

        if height < 2 * self.modes_y:
            raise ValueError("height слишком мал для modes_y.")
        if width // 2 + 1 < self.modes_x:
            raise ValueError("width слишком мал для modes_x.")

        spectrum = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out[:, :, : self.modes_y, : self.modes_x] = self._multiply(
            spectrum[:, :, : self.modes_y, : self.modes_x],
            self.weight_top,
        )
        out[:, :, -self.modes_y :, : self.modes_x] = self._multiply(
            spectrum[:, :, -self.modes_y :, : self.modes_x],
            self.weight_bottom,
        )

        return torch.fft.irfft2(
            out,
            s=(height, width),
            norm="ortho",
        )


class FNO2d(nn.Module):
    """Четырёхслойный FNO с точным нулём на закреплённой левой границе."""

    def __init__(
        self,
        *,
        in_channels: int = INPUT_CHANNELS,
        out_channels: int = OUTPUT_CHANNELS,
        width: int = 24,
        modes_y: int = 8,
        modes_x: int = 16,
        layers: int = 4,
        padding_y: int = 6,
        padding_x: int = 6,
    ) -> None:
        super().__init__()

        self.width = width
        self.modes_y = modes_y
        self.modes_x = modes_x
        self.layers = layers
        self.padding_y = padding_y
        self.padding_x = padding_x

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)

        self.spectral = nn.ModuleList(
            [
                SpectralConv2d(
                    width,
                    width,
                    modes_y,
                    modes_x,
                )
                for _ in range(layers)
            ]
        )
        self.local = nn.ModuleList(
            [
                nn.Conv2d(width, width, kernel_size=1)
                for _ in range(layers)
            ]
        )

        self.project_hidden = nn.Conv2d(width, 64, kernel_size=1)
        self.project_out = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = x.shape[-2]
        width = x.shape[-1]

        x = self.lift(x)
        x = F.pad(
            x,
            (0, self.padding_x, 0, self.padding_y),
        )

        for index, (spectral, local) in enumerate(
            zip(self.spectral, self.local)
        ):
            x = spectral(x) + local(x)
            if index + 1 != self.layers:
                x = F.gelu(x)

        x = x[..., :height, :width]
        x = F.gelu(self.project_hidden(x))
        x = self.project_out(x)

        # Cantilever family: обе компоненты U точно равны нулю при x=0.
        free_x_mask = (
            torch.arange(
                width,
                device=x.device,
            )
            .gt(0)
            .to(dtype=x.dtype)
            .view(1, 1, 1, width)
        )

        return x * free_x_mask


def count_parameters(model: nn.Module) -> int:
    """Число действительных обучаемых скаляров.

    Complex parameter учитывается как два действительных числа.
    """

    total = 0
    for parameter in model.parameters():
        factor = 2 if parameter.is_complex() else 1
        total += factor * parameter.numel()
    return int(total)
