from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biotransport_lab.api import (  # noqa: E402
    simulate_cartesian_preset,
    simulate_radial_preset,
    write_cartesian_csv,
    write_radial_csv,
    write_radial_profiles_csv,
    write_random_walk_csv,
)
from biotransport_lab.random_walk import simulate_random_walk  # noqa: E402
from biotransport_lab.visualization import (  # noqa: E402
    ensure_outdir,
    save_cartesian_outputs,
    save_radial_outputs,
    save_random_walk_outputs,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a gallery of teaching examples.")
    parser.add_argument("--outdir", default="outputs/gallery")
    args = parser.parse_args()
    base = ensure_outdir(args.outdir)

    random_out = ensure_outdir(base / "random_walk_diffusion")
    random_result = simulate_random_walk(particles=1200, steps=200, D_um2_s=80.0, dt_s=0.01)
    save_random_walk_outputs(random_result, random_out)
    write_random_walk_csv(random_result, random_out / "msd.csv")
    write_json(random_result.to_json_summary(), random_out / "statistics.json")

    for preset in [
        "ficks_law_diffusion",
        "microchannel_biosensor",
        "reaction_diffusion_sink",
        "chemotactic_gradient_demo",
    ]:
        out = ensure_outdir(base / preset)
        result = simulate_cartesian_preset(preset=preset, total_time_s=0.8)
        save_cartesian_outputs(result, out)
        write_cartesian_csv(result, out / "diagnostics.csv")
        write_json(result.to_json_summary(), out / "statistics.json")

    radial_examples = [
        ("spherical_cell_uptake", "spherical", "absorbing"),
        ("drug_release_sphere", "spherical", "absorbing"),
        ("cylindrical_vessel_transport", "cylindrical", "noflux"),
    ]
    for preset, geometry, boundary in radial_examples:
        out = ensure_outdir(base / preset)
        result = simulate_radial_preset(
            preset=preset,
            geometry=geometry,
            boundary_kind=boundary,
            total_time_s=0.2,
        )
        save_radial_outputs(result, out)
        write_radial_csv(result, out / "diagnostics.csv")
        write_radial_profiles_csv(result, out / "profiles.csv")
        write_json(result.to_json_summary(), out / "statistics.json")

    print(f"Wrote gallery outputs to {base}")


if __name__ == "__main__":
    main()
