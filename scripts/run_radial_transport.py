from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biotransport_lab.api import (  # noqa: E402
    simulate_radial_preset,
    write_radial_csv,
    write_radial_profiles_csv,
)
from biotransport_lab.visualization import ensure_outdir, save_radial_outputs, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a radial biological transport demo.")
    parser.add_argument("--geometry", choices=["cylindrical", "spherical"], default="spherical")
    parser.add_argument("--preset", default="spherical_cell_uptake")
    parser.add_argument("--D", type=float, default=100.0, help="Diffusion coefficient in um^2/s.")
    parser.add_argument("--k", type=float, default=0.0, help="Reaction rate in 1/s.")
    parser.add_argument("--radius", type=float, default=20.0, help="Outer radius in um.")
    parser.add_argument("--outer-concentration", type=float, default=1.0)
    parser.add_argument("--boundary", choices=["absorbing", "noflux", "fixed"], default="absorbing")
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--outdir", default="outputs/radial")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)
    result = simulate_radial_preset(
        preset=args.preset,
        geometry=args.geometry,
        D_um2_s=args.D,
        k_s=args.k,
        radius_um=args.radius,
        outer_concentration=args.outer_concentration,
        boundary_kind=args.boundary,
        total_time_s=args.total_time,
    )
    save_radial_outputs(result, outdir)
    write_radial_csv(result, outdir / "diagnostics.csv")
    write_radial_profiles_csv(result, outdir / "profiles.csv")
    write_json(result.to_json_summary(), outdir / "statistics.json")
    print(f"Wrote radial outputs to {outdir}")


if __name__ == "__main__":
    main()
