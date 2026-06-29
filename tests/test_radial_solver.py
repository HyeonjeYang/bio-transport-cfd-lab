import numpy as np

from biotransport_lab.core import (
    BoundaryConfig,
    RadialDomain,
    SensorConfig,
    SimulationConfig,
    SourceConfig,
    TransportParameters,
)
from biotransport_lab.solver_radial import simulate_radial_transport, uniform_radial_initial


def test_radial_solver_shape_and_stability():
    domain = RadialDomain(radius_um=20.0, nr=41, geometry="spherical")
    result = simulate_radial_transport(
        domain=domain,
        params=TransportParameters(D_um2_s=80.0, U_um_s=0.0, k_s=0.01),
        config=SimulationConfig(total_time_s=0.05, save_frames=4),
        source=SourceConfig(strength_conc_s=0.0),
        sensor=SensorConfig(absorption_rate_s=0.0),
        boundary=BoundaryConfig(radial_outer="noflux"),
        initial_concentration=uniform_radial_initial(domain, 1.0),
    )
    assert result.profiles.shape == (4, domain.nr)
    assert np.min(result.profiles) >= 0.0
    assert np.isfinite(result.profiles).all()


def test_center_symmetry_for_cylindrical_and_spherical():
    for geometry in ("cylindrical", "spherical"):
        domain = RadialDomain(radius_um=10.0, nr=31, geometry=geometry)
        result = simulate_radial_transport(
            domain=domain,
            params=TransportParameters(D_um2_s=50.0, U_um_s=0.0, k_s=0.0),
            config=SimulationConfig(total_time_s=0.02, save_frames=3),
            source=SourceConfig(strength_conc_s=0.0),
            boundary=BoundaryConfig(radial_outer="noflux"),
            initial_concentration=uniform_radial_initial(domain, 1.0),
        )
        np.testing.assert_allclose(result.profiles[-1], 1.0, rtol=1e-4, atol=1e-4)


def test_absorbing_boundary_gives_positive_outward_flux_and_mass_loss():
    domain = RadialDomain(radius_um=12.0, nr=41, geometry="spherical")
    result = simulate_radial_transport(
        domain=domain,
        params=TransportParameters(D_um2_s=40.0, U_um_s=0.0, k_s=0.0),
        config=SimulationConfig(total_time_s=0.05, save_frames=3),
        source=SourceConfig(strength_conc_s=0.0),
        boundary=BoundaryConfig(radial_outer="absorbing"),
        initial_concentration=uniform_radial_initial(domain, 1.0),
    )
    assert result.boundary_flux[-1] > 0.0
    assert result.total_mass[-1] < result.total_mass[0]
    assert result.metadata["flux_sign"] == "Positive boundary flux is outward from r=0 toward r=R."
