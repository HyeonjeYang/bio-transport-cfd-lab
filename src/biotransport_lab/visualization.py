"""Matplotlib output helpers for classroom simulations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from biotransport_lab.random_walk import RandomWalkResult, gaussian_pdf_1d
from biotransport_lab.solver_cartesian import CartesianSimulationResult
from biotransport_lab.solver_radial import RadialSimulationResult


def ensure_outdir(outdir: str | Path) -> Path:
    """Create and return an output directory."""

    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_cartesian_outputs(result: CartesianSimulationResult, outdir: str | Path) -> dict[str, Path]:
    """Save required Cartesian PNG outputs."""

    out = ensure_outdir(outdir)
    mid = len(result.frames) // 2
    paths = {
        "initial": out / "initial.png",
        "mid_time": out / "mid_time.png",
        "final": out / "final.png",
        "time_evolution_panel": out / "time_evolution_panel.png",
        "statistics": out / "statistics.png",
        "velocity": out / "velocity.png",
    }
    save_concentration_heatmap(result, 0, paths["initial"], "Initial concentration")
    save_concentration_heatmap(result, mid, paths["mid_time"], "Intermediate concentration")
    save_concentration_heatmap(
        result,
        len(result.frames) - 1,
        paths["final"],
        f"{result.state_label.capitalize()}",
    )
    save_cartesian_time_evolution(result, paths["time_evolution_panel"])
    save_cartesian_statistics(result, paths["statistics"])
    save_velocity_quiver(result, paths["velocity"])
    return paths


def save_concentration_heatmap(
    result: CartesianSimulationResult, frame_index: int, path: str | Path, title: str
) -> Path:
    """Save a concentration heatmap with units."""

    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    frame = result.frames[frame_index]
    image = ax.imshow(
        frame,
        origin="lower",
        extent=[
            float(result.x_um.min()),
            float(result.x_um.max()),
            float(result.y_um.min()),
            float(result.y_um.max()),
        ],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(f"{title} at t={result.frame_times_s[frame_index]:.2f} s")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Concentration (a.u.)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_velocity_quiver(result: CartesianSimulationResult, path: str | Path) -> Path:
    """Save a sparse velocity vector plot."""

    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    stride_y = max(1, result.u_um_s.shape[0] // 12)
    stride_x = max(1, result.u_um_s.shape[1] // 18)
    ax.quiver(
        result.x_um[::stride_y, ::stride_x],
        result.y_um[::stride_y, ::stride_x],
        result.u_um_s[::stride_y, ::stride_x],
        result.v_um_s[::stride_y, ::stride_x],
        angles="xy",
        scale_units="xy",
        scale=max(float(np.nanmax(np.hypot(result.u_um_s, result.v_um_s))), 1.0) / 10.0,
    )
    ax.set_title("Prescribed velocity field")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_xlim(float(result.x_um.min()), float(result.x_um.max()))
    ax.set_ylim(float(result.y_um.min()), float(result.y_um.max()))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_cartesian_time_evolution(result: CartesianSimulationResult, path: str | Path) -> Path:
    """Save initial, intermediate, and final concentration panels."""

    indices = [0, len(result.frames) // 2, len(result.frames) - 1]
    titles = ["Initial", "Intermediate", "Final"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), constrained_layout=True)
    vmax = float(np.max(result.frames)) if np.max(result.frames) > 0 else 1.0
    for ax, index, title in zip(axes, indices, titles, strict=True):
        image = ax.imshow(
            result.frames[index],
            origin="lower",
            extent=[
                float(result.x_um.min()),
                float(result.x_um.max()),
                float(result.y_um.min()),
                float(result.y_um.max()),
            ],
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{title}\n{result.frame_times_s[index]:.2f} s")
        ax.set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Concentration (a.u.)")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_cartesian_statistics(result: CartesianSimulationResult, path: str | Path) -> Path:
    """Save sensor, mass, flux, and dimensionless number summaries."""

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), constrained_layout=True)
    axes[0, 0].plot(result.diagnostic_times_s, result.sensor_concentration)
    axes[0, 0].set_title("Sensor response")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("concentration (a.u.)")

    axes[0, 1].plot(result.diagnostic_times_s, result.total_mass)
    axes[0, 1].set_title("Total mass")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("mass (a.u. um^2)")

    axes[1, 0].plot(result.diagnostic_times_s, result.outlet_flux, label="outlet")
    axes[1, 0].plot(result.diagnostic_times_s, result.sensor_absorption_flux, label="sensor")
    axes[1, 0].set_title("Flux estimates")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("flux estimate")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    text = _dimensionless_text(result.dimensionless)
    axes[1, 1].text(0.0, 1.0, text, va="top", family="monospace", fontsize=9)
    axes[1, 1].set_title("Dimensionless numbers")

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_radial_outputs(result: RadialSimulationResult, outdir: str | Path) -> dict[str, Path]:
    """Save required radial PNG outputs."""

    out = ensure_outdir(outdir)
    mid = len(result.profiles) // 2
    paths = {
        "initial": out / "initial.png",
        "mid_time": out / "mid_time.png",
        "final": out / "final.png",
        "time_evolution_panel": out / "time_evolution_panel.png",
        "statistics": out / "statistics.png",
    }
    save_radial_profile(result, 0, paths["initial"], "Initial radial profile")
    save_radial_profile(result, mid, paths["mid_time"], "Intermediate radial profile")
    save_radial_profile(result, len(result.profiles) - 1, paths["final"], result.state_label)
    save_radial_time_evolution(result, paths["time_evolution_panel"])
    save_radial_statistics(result, paths["statistics"])
    return paths


def save_radial_profile(
    result: RadialSimulationResult, frame_index: int, path: str | Path, title: str
) -> Path:
    """Save one radial concentration profile."""

    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
    ax.plot(result.r_um, result.profiles[frame_index], linewidth=2)
    ax.set_title(f"{title} at t={result.frame_times_s[frame_index]:.2f} s")
    ax.set_xlabel("radius r (um)")
    ax.set_ylabel("concentration (a.u.)")
    ax.set_ylim(bottom=0.0)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_radial_time_evolution(result: RadialSimulationResult, path: str | Path) -> Path:
    """Save all saved radial profiles on one plot."""

    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    for profile, time_s in zip(result.profiles, result.frame_times_s, strict=True):
        ax.plot(result.r_um, profile, label=f"{time_s:.2f} s")
    ax.set_title("Radial concentration evolution")
    ax.set_xlabel("radius r (um)")
    ax.set_ylabel("concentration (a.u.)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0.0)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_radial_statistics(result: RadialSimulationResult, path: str | Path) -> Path:
    """Save radial mass, flux, and dimensionless summaries."""

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), constrained_layout=True)
    axes[0].plot(result.diagnostic_times_s, result.total_mass)
    axes[0].set_title("Integrated mass")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("mass")

    axes[1].plot(result.diagnostic_times_s, result.boundary_flux)
    axes[1].set_title("Boundary flux")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("flux")

    axes[2].axis("off")
    axes[2].text(
        0.0,
        1.0,
        _dimensionless_text(result.dimensionless),
        va="top",
        family="monospace",
        fontsize=8,
    )
    axes[2].set_title("Dimensionless numbers")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_random_walk_outputs(result: RandomWalkResult, outdir: str | Path) -> dict[str, Path]:
    """Save random walk particle, histogram, and MSD figures."""

    out = ensure_outdir(outdir)
    paths = {
        "particle_cloud": out / "particle_cloud.png",
        "histogram": out / "histogram.png",
        "msd": out / "msd.png",
        "statistics": out / "statistics.png",
    }
    save_particle_cloud(result, paths["particle_cloud"])
    save_random_walk_histogram(result, paths["histogram"])
    save_msd_curve(result, paths["msd"])
    save_msd_curve(result, paths["statistics"])
    return paths


def save_particle_cloud(result: RandomWalkResult, path: str | Path) -> Path:
    """Save final 1D or 2D particle positions."""

    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    if result.positions.shape[1] == 1:
        ax.scatter(result.positions[:, 0], np.zeros(len(result.positions)), s=4, alpha=0.35)
        ax.set_ylabel("marker row")
    else:
        ax.scatter(result.positions[:, 0], result.positions[:, 1], s=4, alpha=0.35)
        ax.set_ylabel("y (um)")
    ax.set_title("Random walk particle cloud")
    ax.set_xlabel("x (um)")
    ax.axvline(0.0, color="black", linewidth=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_random_walk_histogram(result: RandomWalkResult, path: str | Path) -> Path:
    """Save final x-position histogram with a diffusion Gaussian comparison."""

    x = result.positions[:, 0]
    final_time = float(result.times_s[-1])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.hist(x, bins=40, density=True, alpha=0.6, label="particles")
    grid = np.linspace(float(x.min()), float(x.max()), 300)
    D = float(result.metadata["D_um2_s"])
    ax.plot(grid, gaussian_pdf_1d(grid, D, final_time), color="black", label="diffusion PDE")
    ax.set_title("Histogram and diffusion comparison")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("probability density")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def save_msd_curve(result: RandomWalkResult, path: str | Path) -> Path:
    """Save measured and expected MSD curves."""

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.plot(result.times_s, result.msd_um2, label="simulated")
    ax.plot(result.times_s, result.expected_msd_um2, "--", label="theory")
    ax.set_title("Mean squared displacement")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MSD (um^2)")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return Path(path)


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON with stable indentation."""

    output = Path(path)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output


def _dimensionless_text(dimensionless: dict[str, dict[str, float | str]]) -> str:
    lines = []
    for name, item in dimensionless.items():
        value = float(item["value"])
        lines.append(f"{name:13s} {value:9.3g}")
        lines.append(f"  {item['interpretation']}")
    return "\n".join(lines)
