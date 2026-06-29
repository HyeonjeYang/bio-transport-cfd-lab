from __future__ import annotations

import pytest

from biotransport_lab.openfoam_adapter import (
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
    result = create_scalar_transport_case(tmp_path / "demo")
    assert (tmp_path / "demo" / "0").is_dir()
    assert result["message"] == "Created optional case skeleton."
