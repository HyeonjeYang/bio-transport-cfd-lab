from pathlib import Path

from biotransport_lab.api import (
    csv_from_payload,
    get_presets,
    run_preset_for_payload,
    simulate_cartesian_preset,
    simulate_radial_preset,
    write_random_walk_csv,
)
from biotransport_lab.random_walk import simulate_random_walk
from biotransport_lab.visualization import (
    save_cartesian_outputs,
    save_radial_outputs,
    save_random_walk_outputs,
)


def test_api_serialization_payloads():
    presets = get_presets()
    assert any(item["name"] == "microchannel_biosensor" for item in presets)

    payload = run_preset_for_payload(
        {
            "preset": "microchannel_biosensor",
            "D": 80,
            "U": 100,
            "k": 0.01,
            "total_time": 0.05,
        }
    )
    assert payload["kind"] == "cartesian"
    assert "sensor_concentration" in payload["diagnostics"]
    assert "time_s" in csv_from_payload(payload)


def test_png_generation_smoke(tmp_path: Path):
    cartesian = simulate_cartesian_preset(preset="ficks_law_diffusion", total_time_s=0.05)
    cart_paths = save_cartesian_outputs(cartesian, tmp_path / "cartesian")
    assert (tmp_path / "cartesian" / "initial.png").is_file()
    assert cart_paths["statistics"].stat().st_size > 0

    radial = simulate_radial_preset(
        preset="spherical_cell_uptake",
        radius_um=8.0,
        total_time_s=0.05,
    )
    radial_paths = save_radial_outputs(radial, tmp_path / "radial")
    assert radial_paths["final"].stat().st_size > 0

    random = simulate_random_walk(particles=200, steps=20, D_um2_s=10.0, dt_s=0.01)
    random_paths = save_random_walk_outputs(random, tmp_path / "random")
    write_random_walk_csv(random, tmp_path / "random" / "msd.csv")
    assert random_paths["histogram"].stat().st_size > 0
    assert (tmp_path / "random" / "msd.csv").is_file()
