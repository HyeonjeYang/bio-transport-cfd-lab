# bio-transport-cfd-lab

Undergraduate teaching lab for biological transport: diffusion, advection-reaction, radial transport, and microchannel biosensor demos.

This is for classroom learning, not production CFD.

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

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
pytest
python scripts/run_random_walk.py --outdir outputs/random_walk
python scripts/run_cartesian_transport.py --preset microchannel_biosensor --D 80 --U 200 --k 0.02 --outdir outputs/microchannel
python scripts/run_radial_transport.py --geometry spherical --preset spherical_cell_uptake --radius 10 --D 100 --outdir outputs/spherical_cell
python scripts/run_radial_transport.py --geometry cylindrical --preset cylindrical_vessel_transport --radius 20 --D 120 --outdir outputs/cylindrical_vessel
uvicorn app.main:app --reload
```

Notebook:

```bash
jupyter notebook notebooks/BioTransport_CFD_Lab.ipynb
```

OpenFOAM status:

```bash
python -c "from biotransport_lab.openfoam_adapter import get_openfoam_status; print(get_openfoam_status())"
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

## License

Source code is MIT licensed. Documentation, notebooks, and generated educational figures are CC BY 4.0; see `LICENSE-docs.md`.

No AI tool is listed as a co-author or contributor.
