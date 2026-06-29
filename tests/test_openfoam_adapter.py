from __future__ import annotations

import pytest

from biotransport_lab.openfoam_adapter import (
    OpenFOAMCaseConfig,
    create_scalar_transport_case,
    get_openfoam_status,
    run_openfoam_case,
)


def test_openfoam_status_is_json_friendly():
    status = get_openfoam_status()
    assert "installed" in status
    assert "message" in status


def test_openfoam_degrades_gracefully_when_unavailable(tmp_path):
    status = get_openfoam_status()
    if status["installed"]:
        pytest.skip("This test checks the unavailable-backend path.")
    result = run_openfoam_case(tmp_path / "case")
    assert result["started"] is False
    assert "disabled" in result["message"]


def test_openfoam_case_skeleton_creation(tmp_path):
    config = OpenFOAMCaseConfig(D_um2_s=100.0, U_um_s=250.0, total_time_s=0.25)
    result = create_scalar_transport_case(tmp_path / "demo", config)
    assert (tmp_path / "demo" / "0").is_dir()
    assert (tmp_path / "demo" / "system" / "blockMeshDict").is_file()
    assert (tmp_path / "demo" / "0" / "T").is_file()
    assert result["message"] == "Created runnable scalarTransportFoam microchannel case."
    control = (tmp_path / "demo" / "system" / "controlDict").read_text(encoding="utf-8")
    transport = (tmp_path / "demo" / "constant" / "transportProperties").read_text(
        encoding="utf-8"
    )
    velocity = (tmp_path / "demo" / "0" / "U").read_text(encoding="utf-8")
    assert "scalarTransportFoam" in control
    assert "endTime         0.25;" in control
    assert "1.00000000e-10" in transport
    assert "2.50000000e-04" in velocity
