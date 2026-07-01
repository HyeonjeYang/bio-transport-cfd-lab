from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from biotransport_lab.api import csv_from_payload, get_presets, run_preset_for_payload

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

app = FastAPI(title="BioTransport CFD Lab")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class SimulationRequest(BaseModel):
    preset: str = "microchannel_biosensor"
    D: float = Field(80.0, gt=0)
    U: float = Field(200.0, ge=0)
    k: float = Field(0.02, ge=0)
    source_x: float = 24.0
    source_y: float = 45.0
    sensor_x: float = 170.0
    sensor_y: float = 45.0
    geometry: str = "spherical"
    radius: float = Field(20.0, gt=0)
    outer_concentration: float = Field(1.0, ge=0)
    boundary: str = "absorbing"
    total_time: float = Field(1.0, gt=0)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/presets")
def api_presets() -> list[dict[str, str]]:
    return get_presets()


@app.post("/api/simulate")
def api_simulate(request: SimulationRequest) -> dict[str, Any]:
    return run_preset_for_payload(request.model_dump())


@app.post("/api/export_csv")
def api_export_csv(request: SimulationRequest) -> Response:
    payload = run_preset_for_payload(request.model_dump())
    csv_text = csv_from_payload(payload)
    headers = {"Content-Disposition": f'attachment; filename="{request.preset}.csv"'}
    return Response(content=csv_text, media_type="text/csv", headers=headers)
