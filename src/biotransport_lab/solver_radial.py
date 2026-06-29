"""1D radial diffusion-reaction solvers for cylindrical and spherical examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biotransport_lab.core import (
    BoundaryConfig,
    RadialDomain,
    SensorConfig,
    SimulationConfig,
    SourceConfig,
    TransportParameters,
    to_metadata_dict,
)
from biotransport_lab.dimensionless import summarize_dimensionless_numbers


@dataclass
class RadialSimulationResult:
    """Saved radial profiles and diagnostics."""

    r_um: NDArray[np.float64]
    profiles: NDArray[np.float64]
    frame_times_s: NDArray[np.float64]
    diagnostic_times_s: NDArray[np.float64]
    total_mass: NDArray[np.float64]
    boundary_flux: NDArray[np.float64]
    dimensionless: dict[str, dict[str, float | str]]
    metadata: dict[str, object]
    state_label: str
    warnings: list[str]

    def to_json_summary(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics without full profile arrays."""

        return {
            "r_um": self.r_um.tolist(),
            "frame_times_s": self.frame_times_s.tolist(),
            "diagnostic_times_s": self.diagnostic_times_s.tolist(),
            "total_mass": self.total_mass.tolist(),
            "boundary_flux": self.boundary_flux.tolist(),
            "dimensionless": self.dimensionless,
            "metadata": self.metadata,
            "state_label": self.state_label,
            "warnings": self.warnings,
        }


@dataclass
class _RadialWorkspace:
    """Reusable arrays for one explicit radial update."""

    laplacian: NDArray[np.float64]
    rate: NDArray[np.float64]
    scratch: NDArray[np.float64]

    @classmethod
    def empty_like(cls, concentration: NDArray[np.float64]) -> _RadialWorkspace:
        return cls(*(np.empty_like(concentration) for _ in range(3)))


def radial_grid(domain: RadialDomain) -> NDArray[np.float64]:
    """Return radial grid points in um."""

    return np.linspace(0.0, domain.radius_um, domain.nr)


def uniform_radial_initial(domain: RadialDomain, concentration: float = 1.0) -> NDArray[np.float64]:
    """Return a uniform radial concentration profile."""

    if concentration < 0:
        raise ValueError("Initial concentration cannot be negative.")
    return np.full(domain.nr, concentration, dtype=float)


def gaussian_radial_initial(
    domain: RadialDomain, *, center_r_um: float = 0.0, sigma_um: float = 5.0, amplitude: float = 1.0
) -> NDArray[np.float64]:
    """Return a radial Gaussian source profile."""

    r = radial_grid(domain)
    return amplitude * np.exp(-((r - center_r_um) ** 2) / (2.0 * sigma_um**2))


def simulate_radial_transport(
    *,
    domain: RadialDomain | None = None,
    params: TransportParameters | None = None,
    config: SimulationConfig | None = None,
    source: SourceConfig | None = None,
    sensor: SensorConfig | None = None,
    boundary: BoundaryConfig | None = None,
    initial_concentration: NDArray[np.float64] | float | None = None,
) -> RadialSimulationResult:
    """Solve radial diffusion-reaction in cylindrical or spherical coordinates."""

    domain = RadialDomain() if domain is None else domain
    params = TransportParameters() if params is None else params
    config = SimulationConfig() if config is None else config
    source = SourceConfig() if source is None else source
    sensor = SensorConfig(r_um=domain.radius_um) if sensor is None else sensor
    boundary = BoundaryConfig() if boundary is None else boundary

    r = radial_grid(domain)
    concentration = _prepare_initial(domain, initial_concentration)
    _apply_outer_boundary_inplace(concentration, domain, params, boundary)
    next_concentration = np.empty_like(concentration)
    dt_s, warnings = _stable_radial_timestep(domain, params, config)
    steps = max(1, int(np.ceil(config.total_time_s / dt_s)))
    dt_s = config.total_time_s / steps
    if steps > config.max_steps:
        raise ValueError("Stable timestep requires more steps than max_steps allows.")

    source_mask = np.abs(r - source.r_um) <= source.radius_um
    source_term = np.zeros_like(concentration)
    if source.strength_conc_s != 0.0:
        source_term[source_mask] = source.strength_conc_s
    source_has_value = bool(np.any(source_term))
    sensor_mask = np.zeros_like(concentration, dtype=bool)
    if sensor.r_um is not None and sensor.absorption_rate_s > 0:
        sensor_mask = np.abs(r - sensor.r_um) <= sensor.radius_um
    sink_coefficient = np.full_like(concentration, params.k_s)
    if sensor.absorption_rate_s > 0 and np.any(sensor_mask):
        sink_coefficient[sensor_mask] += sensor.absorption_rate_s
    lower_coeff, upper_coeff = _radial_laplacian_coefficients(r, domain)
    workspace = _RadialWorkspace.empty_like(concentration)
    diagnostic_stride = _diagnostic_stride(steps, config.max_diagnostic_points)
    save_times = np.linspace(0.0, config.total_time_s, config.save_frames)
    next_save = 1
    frames = [concentration.copy()]
    frame_times = [0.0]
    diagnostic_times = [0.0]
    mass_series = [_radial_mass(concentration, r, domain.geometry)]
    flux_series = [_outer_flux(concentration, domain, params)]

    time_s = 0.0
    for step_index in range(1, steps + 1):
        _advance_radial(
            concentration,
            domain,
            params,
            source,
            source_term,
            source_has_value,
            sink_coefficient,
            lower_coeff,
            upper_coeff,
            boundary,
            time_s,
            dt_s,
            workspace,
            next_concentration,
        )
        concentration, next_concentration = next_concentration, concentration
        time_s += dt_s

        min_value = float(concentration.min())
        if min_value < 0.0:
            if min_value < -1e-8:
                warnings.append("Negative concentrations were clipped after an explicit radial step.")
            np.maximum(concentration, 0.0, out=concentration)

        if step_index == steps or step_index % diagnostic_stride == 0:
            diagnostic_times.append(time_s)
            mass_series.append(_radial_mass(concentration, r, domain.geometry))
            flux_series.append(_outer_flux(concentration, domain, params))

        while next_save < len(save_times) and time_s >= save_times[next_save] - 0.5 * dt_s:
            frames.append(concentration.copy())
            frame_times.append(float(save_times[next_save]))
            next_save += 1

    while len(frames) < len(save_times):
        frames.append(concentration.copy())
        frame_times.append(float(save_times[len(frames) - 1]))

    profiles = np.asarray(frames, dtype=float)
    metadata = to_metadata_dict(domain, params, config, source, sensor, boundary)
    metadata.update(
        {
            "dt_s": dt_s,
            "steps": steps,
            "diagnostic_stride": diagnostic_stride,
            "diagnostic_points": len(diagnostic_times),
            "tau_diffusion_s": domain.radius_um**2 / params.D_um2_s,
            "flux_sign": "Positive boundary flux is outward from r=0 toward r=R.",
            "units": {
                "radius": "um",
                "time": "s",
                "D": "um^2/s",
                "k": "1/s",
                "spherical_flux": "concentration*um^3/s",
                "cylindrical_flux": "concentration*um^2/s per unit length",
            },
        }
    )
    dimensionless = summarize_dimensionless_numbers(
        U_um_s=max(params.U_um_s, 1e-12),
        L_um=domain.radius_um,
        D_um2_s=params.D_um2_s,
        k_s=params.k_s,
        rho_kg_m3=params.rho_kg_m3,
        mu_pa_s=params.mu_pa_s,
        surface_rate_um_s=boundary.surface_reaction_um_s,
    )

    return RadialSimulationResult(
        r_um=r,
        profiles=profiles,
        frame_times_s=np.asarray(frame_times, dtype=float),
        diagnostic_times_s=np.asarray(diagnostic_times, dtype=float),
        total_mass=np.asarray(mass_series, dtype=float),
        boundary_flux=np.asarray(flux_series, dtype=float),
        dimensionless=dimensionless,
        metadata=metadata,
        state_label=_steady_state_label(profiles, config.steady_tolerance),
        warnings=warnings,
    )


def _prepare_initial(
    domain: RadialDomain, initial_concentration: NDArray[np.float64] | float | None
) -> NDArray[np.float64]:
    if initial_concentration is None:
        return np.zeros(domain.nr, dtype=float)
    if isinstance(initial_concentration, int | float):
        return uniform_radial_initial(domain, float(initial_concentration))
    concentration = np.asarray(initial_concentration, dtype=float)
    if concentration.shape != (domain.nr,):
        raise ValueError(f"Initial concentration shape must be {(domain.nr,)}.")
    if np.any(concentration < 0):
        raise ValueError("Initial concentration cannot be negative.")
    return concentration.copy()


def _stable_radial_timestep(
    domain: RadialDomain, params: TransportParameters, config: SimulationConfig
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    dimension = 2 if domain.geometry == "cylindrical" else 3
    diffusion_dt = config.diffusion_safety * domain.dr_um**2 / (2.0 * dimension * params.D_um2_s)
    reaction_dt = 0.5 / params.k_s if params.k_s > 0 else float("inf")
    stable_dt = min(diffusion_dt, reaction_dt, config.total_time_s)
    if config.dt_s is not None:
        if config.dt_s > stable_dt:
            warnings.append(
                f"Requested dt={config.dt_s:g} s exceeded stability limit; using {stable_dt:g} s."
            )
            return stable_dt, warnings
        return config.dt_s, warnings
    return stable_dt, warnings


def _advance_radial(
    concentration: NDArray[np.float64],
    domain: RadialDomain,
    params: TransportParameters,
    source: SourceConfig,
    source_term: NDArray[np.float64],
    source_has_value: bool,
    sink_coefficient: NDArray[np.float64],
    lower_coeff: NDArray[np.float64],
    upper_coeff: NDArray[np.float64],
    boundary: BoundaryConfig,
    time_s: float,
    dt_s: float,
    workspace: _RadialWorkspace,
    out: NDArray[np.float64],
) -> None:
    _apply_outer_boundary_inplace(concentration, domain, params, boundary)
    inv_dr2 = 1.0 / domain.dr_um**2
    dimension = 2.0 if domain.geometry == "cylindrical" else 3.0

    workspace.laplacian.fill(0.0)
    workspace.laplacian[0] = 2.0 * dimension * (concentration[1] - concentration[0]) * inv_dr2
    np.multiply(lower_coeff, concentration[:-2], out=workspace.laplacian[1:-1])
    workspace.laplacian[1:-1] -= 2.0 * inv_dr2 * concentration[1:-1]
    workspace.laplacian[1:-1] += upper_coeff * concentration[2:]

    np.multiply(workspace.laplacian, params.D_um2_s, out=workspace.rate)
    np.multiply(sink_coefficient, concentration, out=workspace.scratch)
    workspace.rate -= workspace.scratch
    source_is_active = source.start_s <= time_s and (
        source.end_s is None or time_s <= source.end_s
    )
    if source_has_value and source_is_active:
        workspace.rate += source_term

    np.multiply(workspace.rate, dt_s, out=out)
    out += concentration
    _apply_outer_boundary_inplace(out, domain, params, boundary)


def _apply_outer_boundary(
    concentration: NDArray[np.float64],
    domain: RadialDomain,
    params: TransportParameters,
    boundary: BoundaryConfig,
) -> NDArray[np.float64]:
    updated = concentration.copy()
    _apply_outer_boundary_inplace(updated, domain, params, boundary)
    return updated


def _apply_outer_boundary_inplace(
    updated: NDArray[np.float64],
    domain: RadialDomain,
    params: TransportParameters,
    boundary: BoundaryConfig,
) -> None:
    if boundary.radial_outer == "fixed":
        updated[-1] = boundary.outer_concentration
    elif boundary.radial_outer == "absorbing":
        updated[-1] = 0.0
    elif boundary.radial_outer == "noflux":
        if boundary.surface_reaction_um_s > 0:
            updated[-1] = updated[-2] / (
                1.0 + boundary.surface_reaction_um_s * domain.dr_um / params.D_um2_s
            )
        else:
            updated[-1] = updated[-2]
    else:
        raise ValueError(f"Unsupported radial outer boundary: {boundary.radial_outer}")


def _outer_flux(
    concentration: NDArray[np.float64], domain: RadialDomain, params: TransportParameters
) -> float:
    gradient = (concentration[-1] - concentration[-2]) / domain.dr_um
    flux_density = -params.D_um2_s * gradient
    if domain.geometry == "spherical":
        return float(4.0 * np.pi * domain.radius_um**2 * flux_density)
    return float(2.0 * np.pi * domain.radius_um * flux_density)


def _radial_mass(
    concentration: NDArray[np.float64], r: NDArray[np.float64], geometry: str
) -> float:
    if geometry == "spherical":
        integrand = 4.0 * np.pi * r**2 * concentration
    elif geometry == "cylindrical":
        integrand = 2.0 * np.pi * r * concentration
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")
    return float(np.trapezoid(integrand, r))


def _steady_state_label(profiles: NDArray[np.float64], tolerance: float) -> str:
    if len(profiles) < 2:
        return "final simulated state"
    numerator = float(np.linalg.norm(profiles[-1] - profiles[-2]))
    denominator = float(np.linalg.norm(profiles[-2]) + 1e-12)
    if numerator / denominator < tolerance:
        return "approximate steady state"
    return "final simulated state"


def _radial_laplacian_coefficients(
    r: NDArray[np.float64], domain: RadialDomain
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dr = domain.dr_um
    inv_dr2 = 1.0 / dr**2
    alpha = 1.0 if domain.geometry == "cylindrical" else 2.0
    drift = alpha / (2.0 * dr * r[1:-1])
    return inv_dr2 - drift, inv_dr2 + drift


def _diagnostic_stride(steps: int, max_points: int | None) -> int:
    if max_points is None or steps + 1 <= max_points:
        return 1
    return max(1, int(np.ceil(steps / (max_points - 1))))
