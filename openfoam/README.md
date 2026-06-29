# Optional OpenFOAM Backend

OpenFOAM is not required for the notebook, scripts, tests, or web interface. The default classroom workflow uses Python solvers with prescribed flow fields for fast feedback.

Use OpenFOAM when geometry, pressure-driven flow, or full CFD detail matters more than slider response time.

## Install

Install OpenFOAM externally for your operating system, then confirm:

```bash
foamVersion
```

The Python adapter checks this command. If it is missing, the web app disables OpenFOAM controls and continues with Python simulations.

## Case Skeleton

The adapter can create a simple case directory under `openfoam/cases/`. It does not make normal tests depend on OpenFOAM and does not run CFD automatically on every browser slider change.

Suggested exercises:

- Pressure-driven microchannel flow.
- Passive scalar transport.
- Source or sink terms through `fvOptions`.
