import numpy as np

from biotransport_lab.random_walk import gaussian_pdf_1d, simulate_random_walk


def test_random_walk_1d_msd_scaling():
    result = simulate_random_walk(
        particles=6000, steps=200, D_um2_s=5.0, dt_s=0.01, dimensions=1, seed=3
    )
    expected = 2.0 * 5.0 * 200 * 0.01
    np.testing.assert_allclose(result.msd_um2[-1], expected, rtol=0.18)


def test_random_walk_2d_msd_scaling():
    result = simulate_random_walk(
        particles=6000, steps=200, D_um2_s=5.0, dt_s=0.01, dimensions=2, seed=4
    )
    expected = 4.0 * 5.0 * 200 * 0.01
    np.testing.assert_allclose(result.msd_um2[-1], expected, rtol=0.18)
    assert result.positions.shape == (6000, 2)


def test_gaussian_pdf_is_finite_and_positive():
    x = np.linspace(-5.0, 5.0, 51)
    pdf = gaussian_pdf_1d(x, D_um2_s=2.0, time_s=1.0)
    assert np.isfinite(pdf).all()
    assert np.all(pdf > 0)
