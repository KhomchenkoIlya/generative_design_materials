"""Компактный conditional DDPM для 48x16 density fields."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


IMAGE_HEIGHT = 16
IMAGE_WIDTH = 48
CONDITION_DIM = 5


def downsample_density_2x(rho: np.ndarray) -> np.ndarray:
    """32x96 -> 16x48 average pooling с точным сохранением среднего."""

    rho = np.asarray(rho, dtype=np.float32)

    if rho.ndim == 2:
        rho = rho[None, ...]
        squeeze = True
    elif rho.ndim == 3:
        squeeze = False
    else:
        raise ValueError("rho должен иметь shape (H,W) или (N,H,W).")

    n, height, width = rho.shape
    if height % 2 or width % 2:
        raise ValueError("обе spatial-размерности должны быть чётными.")

    pooled = rho.reshape(
        n,
        height // 2,
        2,
        width // 2,
        2,
    ).mean(axis=(2, 4))

    if squeeze:
        return pooled[0]
    return pooled


def volume_project(
    rho: np.ndarray,
    target: float,
    *,
    tolerance: float = 1.0e-7,
    max_steps: int = 80,
) -> np.ndarray:
    """Проекция через общий shift: clip(rho + lambda, 0, 1).

    Она сохраняет относительный spatial-порядок значений и доводит среднюю
    плотность до target без нового FEM/SIMP.
    """

    field = np.asarray(rho, dtype=np.float64)

    if not 0.0 < target < 1.0:
        raise ValueError("target volume должен лежать в (0,1).")

    lo = -1.0
    hi = 1.0

    for _ in range(max_steps):
        mid = 0.5 * (lo + hi)
        candidate = np.clip(field + mid, 0.0, 1.0)
        mean = float(np.mean(candidate))

        if abs(mean - target) <= tolerance:
            return candidate.astype(np.float32)

        if mean < target:
            lo = mid
        else:
            hi = mid

    candidate = np.clip(
        field + 0.5 * (lo + hi),
        0.0,
        1.0,
    )
    return candidate.astype(np.float32)


def binarity_score(rho: np.ndarray) -> float:
    """0 для бинарного поля, 1 для поля rho=0.5."""

    rho = np.asarray(rho, dtype=np.float64)
    return float(np.mean(4.0 * rho * (1.0 - rho)))


def condition_matrix(
    condition: np.ndarray,
    progress: np.ndarray,
) -> np.ndarray:
    """[f_vol, y/H, cos(theta), sin(theta), progress]."""

    condition = np.asarray(condition, dtype=np.float32)
    progress = np.asarray(progress, dtype=np.float32)

    if condition.ndim != 2 or condition.shape[1] != 4:
        raise ValueError("condition должен иметь shape (N,4).")
    if progress.shape != (condition.shape[0],):
        raise ValueError("progress должен иметь shape (N,).")

    return np.concatenate(
        (
            condition,
            progress[:, None],
        ),
        axis=1,
    ).astype(np.float32)


def condition_statistics(
    condition: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(condition[train_mask], dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def sinusoidal_embedding(
    timesteps: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Standard sinusoidal timestep embedding."""

    if dim % 2:
        raise ValueError("embedding dim должен быть чётным.")

    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(
            half,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / max(half - 1, 1)
    )
    arguments = timesteps.float()[:, None] * frequencies[None, :]
    return torch.cat(
        (torch.sin(arguments), torch.cos(arguments)),
        dim=1,
    )


class ResidualFiLMBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        groups_in = 8 if in_channels % 8 == 0 else 4
        groups_out = 8 if out_channels % 8 == 0 else 4

        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.embedding = nn.Linear(
            embedding_dim,
            2 * out_channels,
        )
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        scale, shift = self.embedding(embedding).chunk(
            2,
            dim=1,
        )
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None])
        h = h + shift[:, :, None, None]
        h = self.conv2(F.silu(h))

        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """Небольшой denoiser для epsilon-prediction."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        embedding_dim: int = 64,
        condition_dim: int = CONDITION_DIM,
    ) -> None:
        super().__init__()

        self.base_channels = base_channels
        self.embedding_dim = embedding_dim
        self.condition_dim = condition_dim

        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.input_conv = nn.Conv2d(
            1,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.block1 = ResidualFiLMBlock(
            base_channels,
            base_channels,
            embedding_dim,
        )
        self.down1 = nn.Conv2d(
            base_channels,
            2 * base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.block2 = ResidualFiLMBlock(
            2 * base_channels,
            2 * base_channels,
            embedding_dim,
        )

        self.down2 = nn.Conv2d(
            2 * base_channels,
            3 * base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.block3 = ResidualFiLMBlock(
            3 * base_channels,
            3 * base_channels,
            embedding_dim,
        )
        self.mid = ResidualFiLMBlock(
            3 * base_channels,
            3 * base_channels,
            embedding_dim,
        )

        self.up2_conv = nn.Conv2d(
            3 * base_channels,
            2 * base_channels,
            kernel_size=3,
            padding=1,
        )
        self.merge2 = nn.Conv2d(
            4 * base_channels,
            2 * base_channels,
            kernel_size=1,
        )
        self.block_up2 = ResidualFiLMBlock(
            2 * base_channels,
            2 * base_channels,
            embedding_dim,
        )

        self.up1_conv = nn.Conv2d(
            2 * base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )
        self.merge1 = nn.Conv2d(
            2 * base_channels,
            base_channels,
            kernel_size=1,
        )
        self.block_up1 = ResidualFiLMBlock(
            base_channels,
            base_channels,
            embedding_dim,
        )

        self.output_norm = nn.GroupNorm(
            8,
            base_channels,
        )
        self.output_conv = nn.Conv2d(
            base_channels,
            1,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = sinusoidal_embedding(
            timesteps,
            self.embedding_dim,
        )
        embedding = (
            self.time_mlp(time_embedding)
            + self.condition_mlp(condition)
        )

        h1 = self.block1(
            self.input_conv(x),
            embedding,
        )

        h2 = self.block2(
            self.down1(h1),
            embedding,
        )

        h3 = self.block3(
            self.down2(h2),
            embedding,
        )
        h3 = self.mid(h3, embedding)

        u2 = F.interpolate(
            h3,
            size=h2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        u2 = self.up2_conv(u2)
        u2 = self.merge2(
            torch.cat((u2, h2), dim=1)
        )
        u2 = self.block_up2(
            u2,
            embedding,
        )

        u1 = F.interpolate(
            u2,
            size=h1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        u1 = self.up1_conv(u1)
        u1 = self.merge1(
            torch.cat((u1, h1), dim=1)
        )
        u1 = self.block_up1(
            u1,
            embedding,
        )

        return self.output_conv(
            F.silu(self.output_norm(u1))
        )


class DiffusionSchedule:
    def __init__(
        self,
        *,
        steps: int = 100,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        self.steps = steps

        self.beta = torch.linspace(
            beta_start,
            beta_end,
            steps,
            device=device,
            dtype=torch.float32,
        )
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(
            self.alpha,
            dim=0,
        )
        self.alpha_bar_prev = torch.cat(
            (
                torch.ones(
                    1,
                    device=device,
                    dtype=torch.float32,
                ),
                self.alpha_bar[:-1],
            )
        )
        self.posterior_variance = (
            self.beta
            * (1.0 - self.alpha_bar_prev)
            / (1.0 - self.alpha_bar)
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bar[
            timesteps
        ].view(-1, 1, 1, 1)

        return (
            torch.sqrt(alpha_bar) * x0
            + torch.sqrt(1.0 - alpha_bar) * noise
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: torch.Tensor,
        *,
        height: int = IMAGE_HEIGHT,
        width: int = IMAGE_WIDTH,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        batch = condition.shape[0]

        x = torch.randn(
            batch,
            1,
            height,
            width,
            generator=generator,
            device=condition.device,
        )

        for step in reversed(range(self.steps)):
            t = torch.full(
                (batch,),
                step,
                dtype=torch.long,
                device=condition.device,
            )

            predicted_noise = model(
                x,
                t,
                condition,
            )

            beta_t = self.beta[step]
            alpha_t = self.alpha[step]
            alpha_bar_t = self.alpha_bar[step]

            mean = (
                x
                - (
                    beta_t
                    / torch.sqrt(1.0 - alpha_bar_t)
                )
                * predicted_noise
            ) / torch.sqrt(alpha_t)

            if step > 0:
                noise = torch.randn(
                    x.shape,
                    generator=generator,
                    device=x.device,
                    dtype=x.dtype,
                )
                variance = self.posterior_variance[step]
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean

        return torch.clamp(x, -1.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
