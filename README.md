# bio-transport-cfd-lab

Undergraduate teaching lab for biological transport: diffusion, advection-reaction, radial transport, and microchannel biosensor demos.

This is for classroom learning, not production CFD.

이 저장소는 2026 서울대학교 여름학기 `생물물리학입문` 특강을 위한 교육자료입니다.

## Attribution and Use

Ideas, equations, biological examples, and teaching direction: Hyeonje Yang.
Implementation was prepared with help from Codex.

Educational use only. These models are not validated for research, clinical, regulatory, or engineering design use.

## What runs

- Python solvers for fast parameter studies.
- Notebook and scripts that save PNG figures and CSV/JSON statistics.
- FastAPI + HTML interface for sliders and plots.
- Optional OpenFOAM adapter for higher-fidelity CFD checks.

OpenFOAM is optional because it is useful for detailed CFD but too heavy for immediate classroom slider updates. The main workflow uses prescribed flow fields and Python transport solvers.

## Equations

Cartesian transport:

```text
dC/dt + u dC/dx + v dC/dy = D (d2C/dx2 + d2C/dy2) - k C + S
```

Cylindrical radial transport:

```text
dC/dt = D (1/r) d/dr(r dC/dr) - k C + S
```

Spherical radial transport:

```text
dC/dt = D (1/r^2) d/dr(r^2 dC/dr) - k C + S
```

Units use micrometers and seconds: `D` in `um^2/s`, `U` in `um/s`, `k` in `1/s`.

## Quickstart

Windows PowerShell:

```powershell
git clone https://github.com/HyeonjeYang/bio-transport-cfd-lab.git
cd bio-transport-cfd-lab
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements-dev.txt
py -m pytest
py scripts\run_all_examples.py
uvicorn app.main:app --reload
```

macOS:

```bash
git clone https://github.com/HyeonjeYang/bio-transport-cfd-lab.git
cd bio-transport-cfd-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts/run_all_examples.py
uvicorn app.main:app --reload
```

Linux:

```bash
git clone https://github.com/HyeonjeYang/bio-transport-cfd-lab.git
cd bio-transport-cfd-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts/run_all_examples.py
uvicorn app.main:app --reload
```

Example scripts:

```bash
python scripts/run_random_walk.py --outdir outputs/random_walk
python scripts/run_cartesian_transport.py --preset microchannel_biosensor --D 80 --U 200 --k 0.02 --outdir outputs/microchannel
python scripts/run_radial_transport.py --geometry spherical --preset spherical_cell_uptake --radius 10 --D 100 --outdir outputs/spherical_cell
python scripts/run_radial_transport.py --geometry cylindrical --preset cylindrical_vessel_transport --radius 20 --D 120 --outdir outputs/cylindrical_vessel
```

Notebook:

```bash
jupyter notebook notebooks/BioTransport_CFD_Lab.ipynb
```

## Integration

Install into another Python project:

```bash
python -m pip install "bio-transport-cfd-lab @ git+https://github.com/HyeonjeYang/bio-transport-cfd-lab.git"
```

Python call:

```python
from biotransport_lab.api import run_preset_for_payload

payload = run_preset_for_payload({"preset": "microchannel_biosensor", "D": 80, "U": 200, "k": 0.02})
```

Local API call:

```bash
uvicorn app.main:app --reload
curl -X POST http://127.0.0.1:8000/api/simulate -H "Content-Type: application/json" -d '{"preset":"microchannel_biosensor","D":80,"U":200,"k":0.02}'
```

PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/simulate -Method Post -ContentType "application/json" -Body '{"preset":"microchannel_biosensor","D":80,"U":200,"k":0.02}'
```

OpenFOAM can be linked through `scripts/run_openfoam_case.py` or `POST /api/run_openfoam`. The Python solver remains the default fast path.

OpenFOAM status:

```bash
python -c "from biotransport_lab.openfoam_adapter import get_openfoam_status; print(get_openfoam_status())"
```

Optional OpenFOAM run:

```bash
python scripts/run_openfoam_case.py --D 80 --U 200 --total-time 0.2 --outdir openfoam/cases/demo
```

## Examples

- Random walk to diffusion.
- Fickian diffusion and flux.
- Microchannel advection-diffusion.
- Biosensor depletion and sensor response.
- Reaction-diffusion in tissue.
- Cylindrical vessel transport.
- Spherical cell uptake and drug release.
- Chemotactic gradient demo.

## Teaching Plan

- 10 min: random walk and diffusion.
- 10 min: Fick's law and flux.
- 15 min: Peclet number in a microchannel.
- 15 min: cylindrical and spherical transport.
- 15 min: biosensor design exercise.

## Limits

The Python models use prescribed flow, simplified reactions, and low-dimensional grids. They are meant to teach scaling and interpretation. Use OpenFOAM separately when flow physics or geometry detail matters.

Long diagnostic curves are downsampled with `SimulationConfig.max_diagnostic_points` for faster browser and CSV workflows. Saved frames and final values are still kept.

## Model Checks

Default teaching presets are tested for finite, nonnegative concentrations, stable timesteps, radial flux direction, and no-flux diffusion mass conservation. These checks support classroom use, not research validation.

## External Tools

Built with Python, NumPy, Matplotlib, FastAPI, Plotly.js, and pytest. Optional high-fidelity runs use OpenFOAM (`blockMesh`, `scalarTransportFoam`) if installed separately.

## License

Source code is MIT licensed. Documentation, notebooks, and generated educational figures are CC BY 4.0; see `LICENSE-docs.md`.
