"""Random walk simulations that connect particle motion to diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RandomWalkResult:
    """Final particle positions and mean-squared displacement diagnostics."""

    positions: NDArray[np.float64]
    times_s: NDArray[np.float64]
    msd_um2: NDArray[np.float64]
    expected_msd_um2: NDArray[np.float64]
    metadata: dict[str, int | float | str]

    def to_json_summary(self) -> dict[str, object]:
        """Return JSON-friendly MSD diagnostics."""

        return {
            "times_s": self.times_s.tolist(),
            "msd_um2": self.msd_um2.tolist(),
            "expected_msd_um2": self.expected_msd_um2.tolist(),
            "metadata": self.metadata,
        }


def simulate_random_walk(
    *,
    particles: int = 1000,
    steps: int = 200,
    D_um2_s: float = 100.0,
    dt_s: float = 0.01,
    dimensions: int = 2,
    seed: int | None = 7,
) -> RandomWalkResult:
    """Simulate independent Brownian walkers in 1D or 2D."""

    if particles < 1 or steps < 1:
        raise ValueError("particles and steps must be positive.")
    if D_um2_s <= 0 or dt_s <= 0:
        raise ValueError("D_um2_s and dt_s must be positive.")
    if dimensions not in {1, 2}:
        raise ValueError("Only 1D and 2D random walks are supported.")

    rng = np.random.default_rng(seed)
    positions = np.zeros((particles, dimensions), dtype=float)
    times = np.arange(steps + 1, dtype=float) * dt_s
    msd = np.zeros(steps + 1, dtype=float)
    step_std = np.sqrt(2.0 * D_um2_s * dt_s)

    for step in range(1, steps + 1):
        positions += rng.normal(0.0, step_std, size=positions.shape)
        msd[step] = float(np.mean(np.sum(positions**2, axis=1)))

    expected = 2.0 * dimensions * D_um2_s * times
    metadata = {
        "particles": particles,
        "steps": steps,
        "D_um2_s": D_um2_s,
        "dt_s": dt_s,
        "dimensions": dimensions,
        "seed": "None" if seed is None else seed,
        "msd_reference": "1D: <x^2> = 2Dt; 2D: <r^2> = 4Dt.",
    }
    return RandomWalkResult(
        positions=positions,
        times_s=times,
        msd_um2=msd,
        expected_msd_um2=expected,
        metadata=metadata,
    )


def gaussian_pdf_1d(x_um: NDArray[np.float64], D_um2_s: float, time_s: float) -> NDArray[np.float64]:
    """Return the 1D diffusion Green's function for a point release."""

    if D_um2_s <= 0 or time_s <= 0:
        raise ValueError("D_um2_s and time_s must be positive.")
    variance = 2.0 * D_um2_s * time_s
    return np.exp(-(x_um**2) / (2.0 * variance)) / np.sqrt(2.0 * np.pi * variance)
