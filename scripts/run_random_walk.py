from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biotransport_lab.api import write_random_walk_csv  # noqa: E402
from biotransport_lab.random_walk import simulate_random_walk  # noqa: E402
from biotransport_lab.visualization import (  # noqa: E402
    ensure_outdir,
    save_random_walk_outputs,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a random-walk diffusion demo.")
    parser.add_argument("--particles", type=int, default=1200)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--D", type=float, default=100.0, help="Diffusion coefficient in um^2/s.")
    parser.add_argument("--dt", type=float, default=0.01, help="Timestep in s.")
    parser.add_argument("--dimensions", type=int, choices=[1, 2], default=2)
    parser.add_argument("--outdir", default="outputs/random_walk")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    result = simulate_random_walk(
        particles=args.particles,
        steps=args.steps,
        D_um2_s=args.D,
        dt_s=args.dt,
        dimensions=args.dimensions,
    )
    save_random_walk_outputs(result, outdir)
    write_random_walk_csv(result, outdir / "msd.csv")
    write_json(result.to_json_summary(), outdir / "statistics.json")
    print(f"Wrote random-walk outputs to {outdir}")


if __name__ == "__main__":
    main()
