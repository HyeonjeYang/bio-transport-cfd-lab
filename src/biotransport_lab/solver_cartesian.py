"""Educational 2D advection-diffusion-reaction solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from biotransport_lab.core import (
    BoundaryConfig,
    CartesianDomain,
    FlowKind,
    SensorConfig,
    SimulationConfig,
    SourceConfig,
    TransportParameters,
    to_metadata_dict,
)
from biotransport_lab.dimensionless import summarize_dimensionless_numbers
from biotransport_lab.fields import cartesian_grid, prescribed_flow


@dataclass
class CartesianSimulationResult:
    """Saved fields and diagnostics from a Cartesian transport run."""

    x_um: NDArray[np.float64]
    y_um: NDArray[np.float64]
    frames: NDArray[np.float64]
    frame_times_s: NDArray[np.float64]
    u_um_s: NDArray[np.float64]
    v_um_s: NDArray[np.float64]
    diagnostic_times_s: NDArray[np.float64]
    sensor_concentration: NDArray[np.float64]
    total_mass: NDArray[np.float64]
    sensor_absorption_flux: NDArray[np.float64]
    outlet_flux: NDArray[np.float64]
    dimensionless: dict[str, dict[str, float | str]]
    metadata: dict[str, object]
    state_label: str
    warnings: list[str]

    def to_json_summary(self) -> dict[str, object]:
        """Return a compact JSON-friendly summary without full arrays."""

        return {
            "frame_times_s": self.frame_times_s.tolist(),
            "diagnostic_times_s": self.diagnostic_times_s.tolist(),
            "sensor_concentration": self.sensor_concentration.tolist(),
            "total_mass": self.total_mass.tolist(),
            "sensor_absorption_flux": self.sensor_absorption_flux.tolist(),
            "outlet_flux": self.outlet_flux.tolist(),
            "dimensionless": self.dimensionless,
            "metadata": self.metadata,
            "state_label": self.state_label,
            "warnings": self.warnings,
        }


@dataclass
class _CartesianWorkspace:
    """Reusable arrays for one explicit Cartesian update."""

    left: NDArray[np.float64]
    right: NDArray[np.float64]
    down: NDArray[np.float64]
    up: NDArray[np.float64]
    dcdx: NDArray[np.float64]
    dcdy: NDArray[np.float64]
    laplacian: NDArray[np.float64]
    rate: NDArray[np.float64]
    scratch: NDArray[np.float64]

    @classmethod
    def empty_like(cls, concentration: NDArray[np.float64]) -> _CartesianWorkspace:
        return cls(*(np.empty_like(concentration) for _ in range(9)))


def gaussian_initial_condition(
    domain: CartesianDomain,
    *,
    center_x_um: float | None = None,
    center_y_um: float | None = None,
    sigma_um: float = 8.0,
    amplitude: float = 1.0,
) -> NDArray[np.float64]:
    """Return a Gaussian concentration patch for diffusion demos."""

    x, y = cartesian_grid(domain)
    cx = domain.width_um / 2.0 if center_x_um is None else center_x_um
    cy = domain.height_um / 2.0 if center_y_um is None else center_y_um
    return amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma_um**2))


def simulate_cartesian_transport(
    *,
    domain: CartesianDomain | None = None,
    params: TransportParameters | None = None,
    config: SimulationConfig | None = None,
    source: SourceConfig | None = None,
    sensor: SensorConfig | None = None,
    boundary: BoundaryConfig | None = None,
    flow_kind: FlowKind = "uniform",
    initial_concentration: NDArray[np.float64] | None = None,
) -> CartesianSimulationResult:
    """Solve 2D advection-diffusion-reaction transport on a teaching grid."""

    domain = CartesianDomain() if domain is None else domain
    params = TransportParameters() if params is None else params
    config = SimulationConfig() if config is None else config
    source = SourceConfig() if source is None else source
    sensor = SensorConfig() if sensor is None else sensor
    boundary = BoundaryConfig() if boundary is None else boundary

    x, y = cartesian_grid(domain)
    u, v = prescribed_flow(flow_kind, domain, params.U_um_s)
    dt_s, dt_warnings = _stable_timestep(domain, params, config, u, v)
    steps = max(1, int(np.ceil(config.total_time_s / dt_s)))
    dt_s = config.total_time_s / steps
    if steps > config.max_steps:
        raise ValueError("Stable timestep requires more steps than max_steps allows.")

    concentration = _prepare_initial_concentration(domain, initial_concentration)
    _apply_cartesian_boundaries_inplace(concentration, boundary)
    next_concentration = np.empty_like(concentration)

    source_mask = _circle_mask(x, y, source.x_um, source.y_um, source.radius_um)
    sensor_mask = _circle_mask(x, y, sensor.x_um, sensor.y_um, sensor.radius_um)
    sensor_cell_count = int(np.count_nonzero(sensor_mask))
    source_term = np.zeros_like(concentration)
    if source.strength_conc_s != 0.0:
        source_term[source_mask] = source.strength_conc_s
    source_has_value = bool(np.any(source_term))
    sink_coefficient = np.full_like(concentration, params.k_s)
    if sensor.absorption_rate_s > 0 and sensor_cell_count:
        sink_coefficient[sensor_mask] += sensor.absorption_rate_s
    u_forward_mask = u < 0.0
    v_forward_mask = v < 0.0
    has_u_forward = bool(np.any(u_forward_mask))
    has_v_forward = bool(np.any(v_forward_mask))
    workspace = _CartesianWorkspace.empty_like(concentration)
    cell_area = domain.dx_um * domain.dy_um
    diagnostic_stride = _diagnostic_stride(steps, config.max_diagnostic_points)

    save_times = np.linspace(0.0, config.total_time_s, config.save_frames)
    next_save = 1
    frames = [concentration.copy()]
    frame_times = [0.0]
    diagnostic_times = [0.0]
    sensor_series = [_masked_mean(concentration, sensor_mask, sensor_cell_count)]
    mass_series = [_mass(concentration, cell_area)]
    absorption_series = [
        _sensor_absorption(concentration, sensor_mask, sensor, cell_area, sensor_cell_count)
    ]
    outlet_series = [_outlet_flux(concentration, u, params.D_um2_s, domain.dx_um, domain.dy_um)]
    warnings = list(dt_warnings)

    time_s = 0.0
    for step_index in range(1, steps + 1):
        _advance_one_step(
            concentration,
            u,
            v,
            params,
            domain,
            source,
            source_term,
            source_has_value,
            sink_coefficient,
            u_forward_mask,
            v_forward_mask,
            has_u_forward,
            has_v_forward,
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
                warnings.append("Negative concentrations were clipped after an explicit step.")
            np.maximum(concentration, 0.0, out=concentration)

        if step_index == steps or step_index % diagnostic_stride == 0:
            diagnostic_times.append(time_s)
            sensor_series.append(_masked_mean(concentration, sensor_mask, sensor_cell_count))
            mass_series.append(_mass(concentration, cell_area))
            absorption_series.append(
                _sensor_absorption(
                    concentration,
                    sensor_mask,
                    sensor,
                    cell_area,
                    sensor_cell_count,
                )
            )
            outlet_series.append(
                _outlet_flux(concentration, u, params.D_um2_s, domain.dx_um, domain.dy_um)
            )

        while next_save < len(save_times) and time_s >= save_times[next_save] - 0.5 * dt_s:
            frames.append(concentration.copy())
            frame_times.append(float(save_times[next_save]))
            next_save += 1

    while len(frames) < len(save_times):
        frames.append(concentration.copy())
        frame_times.append(float(save_times[len(frames) - 1]))

    frames_array = np.asarray(frames, dtype=float)
    state_label = _steady_state_label(frames_array, config.steady_tolerance)
    dimensionless = summarize_dimensionless_numbers(
        U_um_s=params.U_um_s,
        L_um=domain.width_um,
        D_um2_s=params.D_um2_s,
        k_s=params.k_s,
        rho_kg_m3=params.rho_kg_m3,
        mu_pa_s=params.mu_pa_s,
        surface_rate_um_s=sensor.absorption_rate_s * sensor.radius_um,
    )
    metadata = to_metadata_dict(domain, params, config, source, sensor, boundary)
    metadata.update(
        {
            "flow_kind": flow_kind,
            "dt_s": dt_s,
            "steps": steps,
            "diagnostic_stride": diagnostic_stride,
            "diagnostic_points": len(diagnostic_times),
            "units": {
                "length": "um",
                "time": "s",
                "D": "um^2/s",
                "U": "um/s",
                "k": "1/s",
            },
            "negative_clipping": "Values below zero after explicit steps are clipped and recorded in warnings.",
        }
    )

    return CartesianSimulationResult(
        x_um=x,
        y_um=y,
        frames=frames_array,
        frame_times_s=np.asarray(frame_times, dtype=float),
        u_um_s=u,
        v_um_s=v,
        diagnostic_times_s=np.asarray(diagnostic_times, dtype=float),
        sensor_concentration=np.asarray(sensor_series, dtype=float),
        total_mass=np.asarray(mass_series, dtype=float),
        sensor_absorption_flux=np.asarray(absorption_series, dtype=float),
        outlet_flux=np.asarray(outlet_series, dtype=float),
        dimensionless=dimensionless,
        metadata=metadata,
        state_label=state_label,
        warnings=warnings,
    )


def _stable_timestep(
    domain: CartesianDomain,
    params: TransportParameters,
    config: SimulationConfig,
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    dx = domain.dx_um
    dy = domain.dy_um
    diffusion_limit = 1.0 / (2.0 * params.D_um2_s * (1.0 / dx**2 + 1.0 / dy**2))
    diffusion_dt = config.diffusion_safety * diffusion_limit

    max_u = float(np.max(np.abs(u)))
    max_v = float(np.max(np.abs(v)))
    advective_limits = []
    if max_u > 0:
        advective_limits.append(config.cfl * dx / max_u)
    if max_v > 0:
        advective_limits.append(config.cfl * dy / max_v)
    advection_dt = min(advective_limits) if advective_limits else float("inf")
    reaction_dt = 0.5 / params.k_s if params.k_s > 0 else float("inf")
    stable_dt = min(diffusion_dt, advection_dt, reaction_dt, config.total_time_s)

    if config.dt_s is not None:
        if config.dt_s > stable_dt:
            warnings.append(
                f"Requested dt={config.dt_s:g} s exceeded stability limit; using {stable_dt:g} s."
            )
            return stable_dt, warnings
        return config.dt_s, warnings
    return stable_dt, warnings


def _prepare_initial_concentration(
    domain: CartesianDomain, initial_concentration: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    if initial_concentration is None:
        return np.zeros((domain.ny, domain.nx), dtype=float)
    concentration = np.asarray(initial_concentration, dtype=float)
    expected = (domain.ny, domain.nx)
    if concentration.shape != expected:
        raise ValueError(f"Initial concentration shape must be {expected}.")
    if np.any(concentration < 0):
        raise ValueError("Initial concentration cannot be negative.")
    return concentration.copy()


def _advance_one_step(
    concentration: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    params: TransportParameters,
    domain: CartesianDomain,
    source: SourceConfig,
    source_term: NDArray[np.float64],
    source_has_value: bool,
    sink_coefficient: NDArray[np.float64],
    u_forward_mask: NDArray[np.bool_],
    v_forward_mask: NDArray[np.bool_],
    has_u_forward: bool,
    has_v_forward: bool,
    boundary: BoundaryConfig,
    time_s: float,
    dt_s: float,
    workspace: _CartesianWorkspace,
    out: NDArray[np.float64],
) -> None:
    _apply_cartesian_boundaries_inplace(concentration, boundary)
    _fill_neighbors(
        concentration,
        workspace.left,
        workspace.right,
        workspace.down,
        workspace.up,
    )
    dx = domain.dx_um
    dy = domain.dy_um

    np.subtract(concentration, workspace.left, out=workspace.dcdx)
    workspace.dcdx /= dx
    if has_u_forward:
        workspace.dcdx[u_forward_mask] = (
            workspace.right[u_forward_mask] - concentration[u_forward_mask]
        ) / dx

    np.subtract(concentration, workspace.down, out=workspace.dcdy)
    workspace.dcdy /= dy
    if has_v_forward:
        workspace.dcdy[v_forward_mask] = (
            workspace.up[v_forward_mask] - concentration[v_forward_mask]
        ) / dy

    np.add(workspace.right, workspace.left, out=workspace.laplacian)
    workspace.laplacian -= 2.0 * concentration
    workspace.laplacian /= dx**2
    np.add(workspace.up, workspace.down, out=workspace.scratch)
    workspace.scratch -= 2.0 * concentration
    workspace.scratch /= dy**2
    workspace.laplacian += workspace.scratch

    np.multiply(u, workspace.dcdx, out=workspace.rate)
    workspace.rate *= -1.0
    np.multiply(v, workspace.dcdy, out=workspace.scratch)
    workspace.rate -= workspace.scratch
    workspace.rate += params.D_um2_s * workspace.laplacian
    np.multiply(sink_coefficient, concentration, out=workspace.scratch)
    workspace.rate -= workspace.scratch
    source_is_active = source.start_s <= time_s and (
        source.end_s is None or time_s <= source.end_s
    )
    if source_has_value and source_is_active:
        workspace.rate += source_term

    np.multiply(workspace.rate, dt_s, out=out)
    out += concentration
    _apply_cartesian_boundaries_inplace(out, boundary)


def _fill_neighbors(
    concentration: NDArray[np.float64],
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    down: NDArray[np.float64],
    up: NDArray[np.float64],
) -> None:
    left[:, 1:] = concentration[:, :-1]
    left[:, 0] = concentration[:, 0]
    right[:, :-1] = concentration[:, 1:]
    right[:, -1] = concentration[:, -1]
    down[1:, :] = concentration[:-1, :]
    down[0, :] = concentration[0, :]
    up[:-1, :] = concentration[1:, :]
    up[-1, :] = concentration[-1, :]


def _apply_cartesian_boundaries(
    concentration: NDArray[np.float64], boundary: BoundaryConfig
) -> NDArray[np.float64]:
    updated = concentration.copy()
    _apply_cartesian_boundaries_inplace(updated, boundary)
    return updated


def _apply_cartesian_boundaries_inplace(
    updated: NDArray[np.float64], boundary: BoundaryConfig
) -> None:
    if boundary.inlet_concentration is not None:
        updated[:, 0] = boundary.inlet_concentration
    else:
        updated[:, 0] = updated[:, 1]

    if boundary.outlet in {"zero_gradient", "noflux"}:
        updated[:, -1] = updated[:, -2]

    if boundary.walls == "noflux":
        updated[0, :] = updated[1, :]
        updated[-1, :] = updated[-2, :]
    elif boundary.walls == "absorbing":
        updated[0, :] = 0.0
        updated[-1, :] = 0.0
    elif boundary.walls == "fixed":
        updated[0, :] = boundary.outer_concentration
        updated[-1, :] = boundary.outer_concentration
    else:
        raise ValueError(f"Unsupported wall boundary: {boundary.walls}")


def _circle_mask(
    x: NDArray[np.float64], y: NDArray[np.float64], center_x: float, center_y: float, radius: float
) -> NDArray[np.bool_]:
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def _masked_mean(
    concentration: NDArray[np.float64], mask: NDArray[np.bool_], cell_count: int
) -> float:
    if cell_count == 0:
        return float("nan")
    return float(np.sum(concentration[mask]) / cell_count)


def _mass(concentration: NDArray[np.float64], cell_area: float) -> float:
    return float(np.sum(concentration) * cell_area)


def _sensor_absorption(
    concentration: NDArray[np.float64],
    mask: NDArray[np.bool_],
    sensor: SensorConfig,
    cell_area: float,
    cell_count: int,
) -> float:
    if sensor.absorption_rate_s <= 0 or cell_count == 0:
        return 0.0
    return float(sensor.absorption_rate_s * np.sum(concentration[mask]) * cell_area)


def _outlet_flux(
    concentration: NDArray[np.float64],
    u: NDArray[np.float64],
    D_um2_s: float,
    dx_um: float,
    dy_um: float,
) -> float:
    advective = u[:, -1] * concentration[:, -1]
    diffusive = -D_um2_s * (concentration[:, -1] - concentration[:, -2]) / dx_um
    return float(np.sum((advective + diffusive) * dy_um))


def _steady_state_label(frames: NDArray[np.float64], tolerance: float) -> str:
    if len(frames) < 2:
        return "final simulated state"
    numerator = float(np.linalg.norm(frames[-1] - frames[-2]))
    denominator = float(np.linalg.norm(frames[-2]) + 1e-12)
    if numerator / denominator < tolerance:
        return "approximate steady state"
    return "final simulated state"


def _diagnostic_stride(steps: int, max_points: int | None) -> int:
    if max_points is None or steps + 1 <= max_points:
        return 1
    return max(1, int(np.ceil(steps / (max_points - 1))))
