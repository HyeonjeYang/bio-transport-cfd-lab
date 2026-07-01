# bio-transport-cfd-lab

이 리포지토리는 2026 서울대학교 여름학기 `생물물리학입문` 특강을 위한 교육자료입니다.
(생물물리학입문_4차시 : 생물학의 이동현상)

## Attribution and Use

Ideas, equations, biological examples, and teaching direction: Hyeonje Yang.
Implementation was prepared with help from Codex.

Educational use only. These models are not validated for research, clinical, regulatory, or engineering design use.

## What runs

- Python solvers for fast parameter studies.
- Notebook and scripts that display/save PNG figures and CSV/JSON statistics.
- FastAPI + HTML interface for sliders and plots.
- Optional OpenFOAM adapter for higher-fidelity CFD checks.

## Equations

Cartesian transport:

$$\frac{\partial C}{\partial t} + u \frac{\partial C}{\partial x} + v \frac{\partial C}{\partial y} = D\left(\frac{\partial^2 C}{\partial x^2} + \frac{\partial^2 C}{\partial y^2}\right) - kC + S$$

Cylindrical radial transport:

$$\frac{\partial C}{\partial t} = \frac{D}{r} \frac{\partial}{\partial r}\left(r \frac{\partial C}{\partial r}\right) - kC + S$$

Spherical radial transport:

$$\frac{\partial C}{\partial t} = \frac{D}{r^2} \frac{\partial}{\partial r}\left(r^2 \frac{\partial C}{\partial r}\right) - kC + S$$

Units use micrometers and seconds: $D$ in $\mu m^2/s$, $U$ in $\mu m/s$, $k$ in $s^{-1}$.

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

## Integration

Install into another Python project:

```bash
python -m pip install "bio-transport-cfd-lab @ git+https://github.com/HyeonjeYang/bio-transport-cfd-lab.git"
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

## Limits

The Python models use prescribed flow, simplified reactions, and low-dimensional grids. They are meant to teach scaling and interpretation. Use OpenFOAM separately when flow physics or geometry detail matters.

Long diagnostic curves are downsampled with `SimulationConfig.max_diagnostic_points` for faster browser and CSV workflows. Saved frames and final values are still kept.

## Model Checks

Default teaching presets are tested for finite, nonnegative concentrations, stable timesteps, radial flux direction, and no-flux diffusion mass conservation. These checks support classroom use, not research validation.

## License

Source code is MIT licensed. Documentation, notebooks, and generated educational figures are CC BY 4.0; see `LICENSE-docs.md`.
