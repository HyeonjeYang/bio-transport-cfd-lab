"""Shared preset API for scripts, notebooks, and the FastAPI app."""

from __future__ import annotations

import base64
import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

from biotransport_lab.core import (
    BoundaryConfig,
    CartesianDomain,
    RadialDomain,
    SensorConfig,
    SimulationConfig,
    SourceConfig,
    TransportParameters,
)
from biotransport_lab.solver_cartesian import (
    CartesianSimulationResult,
    gaussian_initial_condition,
    simulate_cartesian_transport,
)
from biotransport_lab.solver_radial import (
    RadialSimulationResult,
    gaussian_radial_initial,
    simulate_radial_transport,
    uniform_radial_initial,
)

PRESETS: dict[str, dict[str, str]] = {
    "random_walk_diffusion": {
        "kind": "random_walk",
        "title": "Random walk diffusion",
        "biology": "Molecular Brownian motion before continuum diffusion.",
    },
    "ficks_law_diffusion": {
        "kind": "cartesian",
        "title": "Fickian diffusion",
        "biology": "A released molecule spreads through a small tissue region.",
    },
    "microchannel_advection_diffusion": {
        "kind": "cartesian",
        "title": "Microchannel advection-diffusion",
        "biology": "A molecular plume moves through a microfluidic assay.",
    },
    "reaction_diffusion_sink": {
        "kind": "cartesian",
        "title": "Reaction-diffusion sink",
        "biology": "Nutrient or ligand is consumed by a localized patch.",
    },
    "microchannel_biosensor": {
        "kind": "cartesian",
        "title": "Microchannel biosensor",
        "biology": "Target molecules advect past an absorbing sensor patch.",
    },
    "spherical_cell_uptake": {
        "kind": "radial",
        "title": "Spherical cell uptake",
        "biology": "A spherical cell or aggregate depletes a diffusing nutrient.",
    },
    "cylindrical_vessel_transport": {
        "kind": "radial",
        "title": "Cylindrical vessel transport",
        "biology": "A cylindrical vessel or fiber supplies surrounding tissue.",
    },
    "drug_release_sphere": {
        "kind": "radial",
        "title": "Drug release sphere",
        "biology": "Drug diffuses out of a spherical bead into the medium.",
    },
    "spherical_drug_release": {
        "kind": "radial",
        "title": "Spherical drug release",
        "biology": "Drug diffuses out of a spherical bead into the medium.",
    },
    "oxygen_diffusion_tissue": {
        "kind": "cartesian",
        "title": "Oxygen diffusion in tissue",
        "biology": "Oxygen spreads while cells consume it.",
    },
    "chemotactic_gradient_demo": {
        "kind": "cartesian",
        "title": "Chemotactic gradient demo",
        "biology": "A source and sink shape a stable concentration cue.",
    },
}


def get_presets() -> list[dict[str, str]]:
    """Return available biological teaching presets."""

    return [{"name": name, **info} for name, info in PRESETS.items()]


def simulate_cartesian_preset(
    *,
    preset: str = "microchannel_biosensor",
    D_um2_s: float = 80.0,
    U_um_s: float = 200.0,
    k_s: float = 0.02,
    source_x_um: float | None = None,
    source_y_um: float | None = None,
    sensor_x_um: float | None = None,
    sensor_y_um: float | None = None,
    total_time_s: float = 1.0,
) -> CartesianSimulationResult:
    """Run a Cartesian preset with classroom-safe defaults."""

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    domain = CartesianDomain(width_um=220.0, height_um=90.0, nx=81, ny=35)
    source_y = domain.height_um / 2.0 if source_y_um is None else source_y_um
    sensor_y = domain.height_um / 2.0 if sensor_y_um is None else sensor_y_um
    source_x = 24.0 if source_x_um is None else source_x_um
    sensor_x = 170.0 if sensor_x_um is None else sensor_x_um
    params = TransportParameters(D_um2_s=D_um2_s, U_um_s=U_um_s, k_s=k_s)
    config = SimulationConfig(total_time_s=total_time_s, save_frames=6)
    source = SourceConfig(x_um=source_x, y_um=source_y, radius_um=6.0, strength_conc_s=3.0)
    sensor = SensorConfig(
        x_um=sensor_x, y_um=sensor_y, radius_um=8.0, absorption_rate_s=0.0
    )
    boundary = BoundaryConfig(walls="noflux", inlet_concentration=None, outlet="zero_gradient")
    flow_kind = "uniform"
    initial = None

    if preset == "ficks_law_diffusion":
        params = TransportParameters(D_um2_s=D_um2_s, U_um_s=0.0, k_s=0.0)
        source = SourceConfig(strength_conc_s=0.0)
        initial = gaussian_initial_condition(domain, center_x_um=70.0, center_y_um=45.0, sigma_um=8.0)
    elif preset == "microchannel_advection_diffusion":
        params = TransportParameters(D_um2_s=D_um2_s, U_um_s=U_um_s, k_s=0.0)
        flow_kind = "poiseuille"
    elif preset == "reaction_diffusion_sink":
        params = TransportParameters(D_um2_s=D_um2_s, U_um_s=0.0, k_s=k_s)
        sensor = SensorConfig(x_um=sensor_x, y_um=sensor_y, radius_um=12.0, absorption_rate_s=0.15)
    elif preset == "microchannel_biosensor":
        flow_kind = "poiseuille"
        sensor = SensorConfig(x_um=sensor_x, y_um=18.0 if sensor_y_um is None else sensor_y, radius_um=8.0, absorption_rate_s=0.2)
    elif preset == "oxygen_diffusion_tissue":
        params = TransportParameters(D_um2_s=D_um2_s, U_um_s=0.0, k_s=max(k_s, 0.01))
        source = SourceConfig(x_um=20.0, y_um=45.0, radius_um=8.0, strength_conc_s=2.0)
    elif preset == "chemotactic_gradient_demo":
        params = TransportParameters(D_um2_s=D_um2_s, U_um_s=0.0, k_s=max(k_s, 0.005))
        source = SourceConfig(x_um=35.0, y_um=45.0, radius_um=8.0, strength_conc_s=2.5)
        sensor = SensorConfig(x_um=185.0, y_um=45.0, radius_um=12.0, absorption_rate_s=0.08)

    return simulate_cartesian_transport(
        domain=domain,
        params=params,
        config=config,
        source=source,
        sensor=sensor,
        boundary=boundary,
        flow_kind=flow_kind,
        initial_concentration=initial,
    )


def simulate_radial_preset(
    *,
    preset: str = "spherical_cell_uptake",
    geometry: str = "spherical",
    D_um2_s: float = 100.0,
    k_s: float = 0.0,
    radius_um: float = 20.0,
    outer_concentration: float = 1.0,
    boundary_kind: str = "absorbing",
    total_time_s: float = 1.0,
) -> RadialSimulationResult:
    """Run a radial biological preset."""

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    if geometry not in {"cylindrical", "spherical"}:
        raise ValueError("geometry must be cylindrical or spherical.")

    domain = RadialDomain(radius_um=radius_um, nr=51, geometry=geometry)  # type: ignore[arg-type]
    params = TransportParameters(D_um2_s=D_um2_s, U_um_s=0.0, k_s=k_s)
    config = SimulationConfig(total_time_s=total_time_s, save_frames=6, max_steps=100000)
    boundary = BoundaryConfig(radial_outer=boundary_kind, outer_concentration=outer_concentration)  # type: ignore[arg-type]
    source = SourceConfig(strength_conc_s=0.0)
    initial: np.ndarray | float | None = uniform_radial_initial(domain, outer_concentration)

    if preset == "spherical_cell_uptake":
        domain = RadialDomain(radius_um=radius_um, nr=51, geometry="spherical")
        boundary = BoundaryConfig(radial_outer=boundary_kind, outer_concentration=outer_concentration)
        initial = uniform_radial_initial(domain, outer_concentration)
    elif preset in {"drug_release_sphere", "spherical_drug_release"}:
        domain = RadialDomain(radius_um=radius_um, nr=51, geometry="spherical")
        boundary = BoundaryConfig(radial_outer="absorbing", outer_concentration=0.0)
        initial = uniform_radial_initial(domain, outer_concentration)
    elif preset == "cylindrical_vessel_transport":
        domain = RadialDomain(radius_um=radius_um, nr=51, geometry="cylindrical")
        boundary = BoundaryConfig(radial_outer=boundary_kind, outer_concentration=0.0)
        source = SourceConfig(r_um=0.0, radius_um=max(radius_um * 0.08, 1.0), strength_conc_s=1.5)
        initial = gaussian_radial_initial(domain, center_r_um=0.0, sigma_um=max(radius_um * 0.10, 1.0), amplitude=outer_concentration)

    return simulate_radial_transport(
        domain=domain,
        params=params,
        config=config,
        source=source,
        boundary=boundary,
        initial_concentration=initial,
    )


def compact_cartesian_payload(result: CartesianSimulationResult) -> dict[str, Any]:
    """Return arrays small enough for browser plotting."""

    return {
        "kind": "cartesian",
        "x_um": result.x_um[0, :].tolist(),
        "y_um": result.y_um[:, 0].tolist(),
        "frames": result.frames.tolist(),
        "frame_times_s": result.frame_times_s.tolist(),
        "diagnostics": result.to_json_summary(),
    }


def compact_radial_payload(result: RadialSimulationResult) -> dict[str, Any]:
    """Return radial arrays for browser plotting."""

    return {
        "kind": "radial",
        "r_um": result.r_um.tolist(),
        "profiles": result.profiles.tolist(),
        "frame_times_s": result.frame_times_s.tolist(),
        "diagnostics": result.to_json_summary(),
    }


def run_preset_for_payload(params: dict[str, Any]) -> dict[str, Any]:
    """Run the requested preset and return a browser-ready payload."""

    preset = str(params.get("preset", "microchannel_biosensor"))
    if preset not in PRESETS:
        preset = "microchannel_biosensor"
    kind = PRESETS.get(preset, PRESETS["microchannel_biosensor"])["kind"]
    if kind == "radial":
        result = simulate_radial_preset(
            preset=preset,
            geometry=str(params.get("geometry", "spherical")),
            D_um2_s=float(params.get("D", 100.0)),
            k_s=float(params.get("k", 0.0)),
            radius_um=float(params.get("radius", 20.0)),
            outer_concentration=float(params.get("outer_concentration", 1.0)),
            boundary_kind=str(params.get("boundary", "absorbing")),
            total_time_s=float(params.get("total_time", 1.0)),
        )
        return compact_radial_payload(result)

    result = simulate_cartesian_preset(
        preset=preset,
        D_um2_s=float(params.get("D", 80.0)),
        U_um_s=float(params.get("U", 200.0)),
        k_s=float(params.get("k", 0.02)),
        source_x_um=float(params.get("source_x", 24.0)),
        source_y_um=float(params.get("source_y", 45.0)),
        sensor_x_um=float(params.get("sensor_x", 170.0)),
        sensor_y_um=float(params.get("sensor_y", 45.0)),
        total_time_s=float(params.get("total_time", 1.0)),
    )
    return compact_cartesian_payload(result)


def csv_from_payload(payload: dict[str, Any]) -> str:
    """Create CSV text from compact simulation payload."""

    output = io.StringIO()
    writer = csv.writer(output)
    diagnostics = payload["diagnostics"]
    if payload["kind"] == "cartesian":
        writer.writerow(["time_s", "sensor_concentration", "total_mass", "outlet_flux"])
        for row in zip(
            diagnostics["diagnostic_times_s"],
            diagnostics["sensor_concentration"],
            diagnostics["total_mass"],
            diagnostics["outlet_flux"],
            strict=True,
        ):
            writer.writerow(row)
    else:
        writer.writerow(["time_s", "total_mass", "boundary_flux"])
        for row in zip(
            diagnostics["diagnostic_times_s"],
            diagnostics["total_mass"],
            diagnostics["boundary_flux"],
            strict=True,
        ):
            writer.writerow(row)
    return output.getvalue()


def figure_png_base64_from_payload(payload: dict[str, Any]) -> str:
    """Render the final simulated field to a base64 PNG string."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
    if payload["kind"] == "cartesian":
        image = ax.imshow(
            np.asarray(payload["frames"][-1]),
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax.set_title("Final concentration")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
        fig.colorbar(image, ax=ax, label="concentration")
    else:
        ax.plot(payload["r_um"], payload["profiles"][-1])
        ax.set_title("Final radial profile")
        ax.set_xlabel("radius (um)")
        ax.set_ylabel("concentration")
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def write_cartesian_csv(result: CartesianSimulationResult, path: str | Path) -> Path:
    """Write Cartesian diagnostic curves to CSV."""

    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["time_s", "sensor_concentration", "total_mass", "sensor_absorption_flux", "outlet_flux"]
        )
        for row in zip(
            result.diagnostic_times_s,
            result.sensor_concentration,
            result.total_mass,
            result.sensor_absorption_flux,
            result.outlet_flux,
            strict=True,
        ):
            writer.writerow(row)
    return output


def write_radial_csv(result: RadialSimulationResult, path: str | Path) -> Path:
    """Write radial diagnostics to CSV."""

    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "total_mass", "boundary_flux"])
        for row in zip(
            result.diagnostic_times_s,
            result.total_mass,
            result.boundary_flux,
            strict=True,
        ):
            writer.writerow(row)
    return output


def write_radial_profiles_csv(result: RadialSimulationResult, path: str | Path) -> Path:
    """Write saved radial profiles to CSV."""

    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["r_um", *[f"t_{time_s:.4g}_s" for time_s in result.frame_times_s]])
        for index, radius in enumerate(result.r_um):
            writer.writerow([radius, *result.profiles[:, index]])
    return output


def write_random_walk_csv(result: Any, path: str | Path) -> Path:
    """Write random walk MSD diagnostics to CSV."""

    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "msd_um2", "expected_msd_um2"])
        for row in zip(result.times_s, result.msd_um2, result.expected_msd_um2, strict=True):
            writer.writerow(row)
    return output
