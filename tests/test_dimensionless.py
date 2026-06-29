import math

import pytest

from biotransport_lab.dimensionless import (
    biot_like_number,
    damkohler_advection,
    damkohler_diffusion,
    peclet_number,
    reynolds_number,
    summarize_dimensionless_numbers,
)


def test_dimensionless_values():
    assert peclet_number(200.0, 100.0, 50.0) == pytest.approx(400.0)
    assert damkohler_advection(0.02, 100.0, 200.0) == pytest.approx(0.01)
    assert damkohler_diffusion(0.02, 100.0, 50.0) == pytest.approx(4.0)
    assert biot_like_number(2.0, 10.0, 100.0) == pytest.approx(0.2)


def test_reynolds_uses_si_conversion():
    re = reynolds_number(1000.0, 1000.0, 100.0, 1e-3)
    assert re == pytest.approx(0.1)
    assert math.isfinite(re)


def test_dimensionless_summary_is_json_friendly():
    summary = summarize_dimensionless_numbers(
        U_um_s=200.0,
        L_um=100.0,
        D_um2_s=80.0,
        k_s=0.02,
        surface_rate_um_s=1.0,
    )
    assert "Pe" in summary
    assert isinstance(summary["Pe"]["value"], float)
    assert isinstance(summary["Pe"]["interpretation"], str)


def test_dimensionless_rejects_bad_inputs():
    with pytest.raises(ValueError):
        peclet_number(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        damkohler_advection(0.1, 10.0, 0.0)
