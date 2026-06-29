from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biotransport_lab.api import simulate_cartesian_preset, write_cartesian_csv  # noqa: E402
from biotransport_lab.visualization import (  # noqa: E402
    ensure_outdir,
    save_cartesian_outputs,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2D biological transport demo.")
    parser.add_argument("--preset", default="microchannel_biosensor")
    parser.add_argument("--D", type=float, default=80.0, help="Diffusion coefficient in um^2/s.")
    parser.add_argument("--U", type=float, default=200.0, help="Velocity scale in um/s.")
    parser.add_argument("--k", type=float, default=0.02, help="Reaction rate in 1/s.")
    parser.add_argument("--source-x", type=float, default=24.0)
    parser.add_argument("--source-y", type=float, default=45.0)
    parser.add_argument("--sensor-x", type=float, default=170.0)
    parser.add_argument("--sensor-y", type=float, default=45.0)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--outdir", default="outputs/cartesian")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    result = simulate_cartesian_preset(
        preset=args.preset,
        D_um2_s=args.D,
        U_um_s=args.U,
        k_s=args.k,
        source_x_um=args.source_x,
        source_y_um=args.source_y,
        sensor_x_um=args.sensor_x,
        sensor_y_um=args.sensor_y,
        total_time_s=args.total_time,
    )
    save_cartesian_outputs(result, outdir)
    write_cartesian_csv(result, outdir / "diagnostics.csv")
    write_json(result.to_json_summary(), outdir / "statistics.json")
    print(f"Wrote Cartesian outputs to {outdir}")


if __name__ == "__main__":
    main()
