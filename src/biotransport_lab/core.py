"""Core data models for biological transport teaching examples.

The default unit system is micrometers and seconds:
length in um, diffusion coefficient in um^2/s, velocity in um/s,
and first-order reaction rate in 1/s.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


BoundaryKind = Literal["noflux", "fixed", "absorbing"]
RadialGeometry = Literal["cylindrical", "spherical"]
FlowKind = Literal["uniform", "poiseuille", "shear", "vortex", "radial"]


@dataclass(frozen=True)
class CartesianDomain:
    """2D rectangular domain.

    Attributes:
        width_um: Domain length along x in um.
        height_um: Domain length along y in um.
        nx: Number of x grid points.
        ny: Number of y grid points.
    """

    width_um: float = 200.0
    height_um: float = 100.0
    nx: int = 61
    ny: int = 31

    def __post_init__(self) -> None:
        if self.width_um <= 0 or self.height_um <= 0:
            raise ValueError("Domain dimensions must be positive.")
        if self.nx < 3 or self.ny < 3:
            raise ValueError("Cartesian grid needs at least three points per direction.")

    @property
    def dx_um(self) -> float:
        return self.width_um / (self.nx - 1)

    @property
    def dy_um(self) -> float:
        return self.height_um / (self.ny - 1)


@dataclass(frozen=True)
class RadialDomain:
    """1D radial domain for cylindrical or spherical transport.

    Attributes:
        radius_um: Outer radius in um.
        nr: Number of radial grid points, including r = 0 and r = R.
        geometry: Coordinate system for radial diffusion.
    """

    radius_um: float = 20.0
    nr: int = 81
    geometry: RadialGeometry = "spherical"

    def __post_init__(self) -> None:
        if self.radius_um <= 0:
            raise ValueError("Radius must be positive.")
        if self.nr < 4:
            raise ValueError("Radial grid needs at least four points.")
        if self.geometry not in {"cylindrical", "spherical"}:
            raise ValueError("Geometry must be 'cylindrical' or 'spherical'.")

    @property
    def dr_um(self) -> float:
        return self.radius_um / (self.nr - 1)


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical run settings.

    Attributes:
        total_time_s: Simulated time in seconds.
        dt_s: Optional requested timestep in seconds. Solvers may reduce it for stability.
        save_frames: Number of saved frames from initial to final state.
        cfl: Safety factor for explicit advection.
        diffusion_safety: Safety factor for explicit diffusion.
        steady_tolerance: Relative late-frame change used for steady-like labels.
        max_steps: Upper step count guard for classroom runs.
    """

    total_time_s: float = 2.0
    dt_s: float | None = None
    save_frames: int = 6
    cfl: float = 0.45
    diffusion_safety: float = 0.20
    steady_tolerance: float = 1e-3
    max_steps: int = 20000

    def __post_init__(self) -> None:
        if self.total_time_s <= 0:
            raise ValueError("Total time must be positive.")
        if self.dt_s is not None and self.dt_s <= 0:
            raise ValueError("Requested timestep must be positive.")
        if self.save_frames < 3:
            raise ValueError("At least three frames are needed.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")


@dataclass(frozen=True)
class TransportParameters:
    """Transport and material parameters.

    Attributes:
        D_um2_s: Diffusion coefficient in um^2/s.
        U_um_s: Characteristic velocity in um/s.
        k_s: First-order decay or reaction rate in 1/s.
        rho_kg_m3: Fluid density for Reynolds number estimates.
        mu_pa_s: Dynamic viscosity for Reynolds number estimates.
    """

    D_um2_s: float = 100.0
    U_um_s: float = 100.0
    k_s: float = 0.0
    rho_kg_m3: float = 1000.0
    mu_pa_s: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.D_um2_s <= 0:
            raise ValueError("Diffusion coefficient must be positive.")
        if self.U_um_s < 0 or self.k_s < 0:
            raise ValueError("Velocity scale and reaction rate cannot be negative.")
        if self.rho_kg_m3 <= 0 or self.mu_pa_s <= 0:
            raise ValueError("Density and viscosity must be positive.")


@dataclass(frozen=True)
class SourceConfig:
    """Localized concentration source.

    Attributes:
        x_um: Source center x position in um.
        y_um: Source center y position in um.
        r_um: Radial source center in um.
        radius_um: Source radius in um.
        strength_conc_s: Added concentration per second inside the source.
        start_s: Source start time in seconds.
        end_s: Optional source end time in seconds.
    """

    x_um: float = 30.0
    y_um: float = 50.0
    r_um: float = 0.0
    radius_um: float = 6.0
    strength_conc_s: float = 2.0
    start_s: float = 0.0
    end_s: float | None = None


@dataclass(frozen=True)
class SensorConfig:
    """Local sensor or absorbing patch.

    Attributes:
        x_um: Sensor center x position in um.
        y_um: Sensor center y position in um.
        r_um: Radial sensor position in um.
        radius_um: Sensor radius in um.
        absorption_rate_s: Optional first-order sink rate inside the patch in 1/s.
    """

    x_um: float = 160.0
    y_um: float = 50.0
    r_um: float | None = None
    radius_um: float = 8.0
    absorption_rate_s: float = 0.0


@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary condition choices for classroom models.

    Attributes:
        walls: Top and bottom Cartesian wall condition.
        inlet_concentration: Optional inlet Dirichlet concentration.
        outlet: Cartesian outlet condition; currently zero-gradient or no-flux style.
        radial_outer: Outer radial boundary type.
        outer_concentration: Fixed outer concentration when radial_outer is fixed.
        surface_reaction_um_s: Surface reaction velocity in um/s for reactive boundaries.
    """

    walls: BoundaryKind = "noflux"
    inlet_concentration: float | None = None
    outlet: Literal["zero_gradient", "noflux"] = "zero_gradient"
    radial_outer: BoundaryKind = "noflux"
    outer_concentration: float = 0.0
    surface_reaction_um_s: float = 0.0


def to_metadata_dict(*objects: object) -> dict[str, object]:
    """Convert dataclass objects into a single JSON-friendly metadata dictionary."""

    metadata: dict[str, object] = {}
    for obj in objects:
        if hasattr(obj, "__dataclass_fields__"):
            metadata[obj.__class__.__name__] = asdict(obj)
        else:
            metadata[obj.__class__.__name__] = str(obj)
    return metadata
