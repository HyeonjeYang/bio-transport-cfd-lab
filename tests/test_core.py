import pytest

from biotransport_lab.core import CartesianDomain, RadialDomain, SimulationConfig


def test_cartesian_spacing():
    domain = CartesianDomain(width_um=100.0, height_um=50.0, nx=11, ny=6)
    assert domain.dx_um == pytest.approx(10.0)
    assert domain.dy_um == pytest.approx(10.0)


def test_radial_spacing_and_validation():
    domain = RadialDomain(radius_um=20.0, nr=11, geometry="spherical")
    assert domain.dr_um == pytest.approx(2.0)
    with pytest.raises(ValueError):
        RadialDomain(radius_um=0.0)


def test_simulation_config_validation():
    config = SimulationConfig(total_time_s=1.0, save_frames=4)
    assert config.total_time_s == 1.0
    with pytest.raises(ValueError):
        SimulationConfig(total_time_s=-1.0)
    with pytest.raises(ValueError):
        SimulationConfig(max_diagnostic_points=1)
