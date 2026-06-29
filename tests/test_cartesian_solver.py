import numpy as np

from biotransport_lab.core import (
    BoundaryConfig,
    CartesianDomain,
    SensorConfig,
    SimulationConfig,
    SourceConfig,
    TransportParameters,
)
from biotransport_lab.fields import prescribed_flow
from biotransport_lab.solver_cartesian import (
    gaussian_initial_condition,
    simulate_cartesian_transport,
)


def test_prescribed_flow_shapes():
    domain = CartesianDomain(nx=12, ny=8)
    u, v = prescribed_flow("poiseuille", domain, 100.0)
    assert u.shape == (domain.ny, domain.nx)
    assert v.shape == (domain.ny, domain.nx)
    assert np.max(u) > np.mean(u)


def test_diffusion_mass_conservation_with_noflux_boundaries():
    domain = CartesianDomain(width_um=80.0, height_um=80.0, nx=25, ny=25)
    initial = gaussian_initial_condition(domain, sigma_um=8.0)
    result = simulate_cartesian_transport(
        domain=domain,
        params=TransportParameters(D_um2_s=30.0, U_um_s=0.0, k_s=0.0),
        config=SimulationConfig(total_time_s=0.2, save_frames=4),
        source=SourceConfig(strength_conc_s=0.0),
        sensor=SensorConfig(absorption_rate_s=0.0),
        boundary=BoundaryConfig(walls="noflux", outlet="noflux"),
        flow_kind="uniform",
        initial_concentration=initial,
    )
    np.testing.assert_allclose(result.total_mass[-1], result.total_mass[0], rtol=0.02)


def test_cartesian_solver_nonnegative_and_serializable():
    domain = CartesianDomain(width_um=80.0, height_um=40.0, nx=21, ny=13)
    result = simulate_cartesian_transport(
        domain=domain,
        params=TransportParameters(D_um2_s=40.0, U_um_s=60.0, k_s=0.01),
        config=SimulationConfig(total_time_s=0.1, save_frames=3),
        source=SourceConfig(x_um=10.0, y_um=20.0, strength_conc_s=1.0),
        sensor=SensorConfig(x_um=60.0, y_um=20.0, absorption_rate_s=0.1),
        flow_kind="poiseuille",
    )
    assert result.frames.shape == (3, domain.ny, domain.nx)
    assert np.min(result.frames) >= 0.0
    summary = result.to_json_summary()
    assert summary["state_label"] in {"approximate steady state", "final simulated state"}
    assert "Pe" in summary["dimensionless"]


def test_cartesian_diagnostics_can_be_downsampled():
    domain = CartesianDomain(width_um=80.0, height_um=40.0, nx=21, ny=13)
    config = SimulationConfig(total_time_s=0.2, save_frames=3, max_diagnostic_points=5)
    result = simulate_cartesian_transport(
        domain=domain,
        params=TransportParameters(D_um2_s=40.0, U_um_s=60.0, k_s=0.01),
        config=config,
        source=SourceConfig(x_um=10.0, y_um=20.0, strength_conc_s=1.0),
        sensor=SensorConfig(x_um=60.0, y_um=20.0, absorption_rate_s=0.1),
        flow_kind="poiseuille",
    )

    assert len(result.diagnostic_times_s) <= config.max_diagnostic_points
    assert result.diagnostic_times_s[0] == 0.0
    assert np.isclose(result.diagnostic_times_s[-1], config.total_time_s)
