from __future__ import annotations

import pytest


def test_fastapi_endpoints_smoke():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    presets = client.get("/api/presets")
    assert presets.status_code == 200
    assert any(item["name"] == "microchannel_biosensor" for item in presets.json())

    response = client.post(
        "/api/simulate",
        json={
            "preset": "microchannel_biosensor",
            "D": 80,
            "U": 120,
            "k": 0.01,
            "total_time": 0.05,
        },
    )
    assert response.status_code == 200
    assert response.json()["kind"] == "cartesian"

    status = client.get("/api/openfoam/status")
    assert status.status_code == 200
    assert "installed" in status.json()
