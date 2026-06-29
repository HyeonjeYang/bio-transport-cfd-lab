from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biotransport_lab.openfoam_adapter import (  # noqa: E402
    OpenFOAMCaseConfig,
    run_openfoam_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional OpenFOAM scalar case.")
    parser.add_argument("--D", type=float, default=80.0, help="Diffusion coefficient in um^2/s.")
    parser.add_argument("--U", type=float, default=200.0, help="Velocity scale in um/s.")
    parser.add_argument("--total-time", type=float, default=0.2, help="Simulation time in s.")
    parser.add_argument("--nx", type=int, default=44)
    parser.add_argument("--ny", type=int, default=18)
    parser.add_argument("--outdir", default="openfoam/cases/demo")
    args = parser.parse_args()

    config = OpenFOAMCaseConfig(
        D_um2_s=args.D,
        U_um_s=args.U,
        nx=args.nx,
        ny=args.ny,
        total_time_s=args.total_time,
        write_interval_s=max(args.total_time / 2.0, 0.01),
    )
    result = run_openfoam_case(args.outdir, config=config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
