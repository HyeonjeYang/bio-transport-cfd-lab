from __future__ import annotations

import runpy
import sys
from pathlib import Path


def test_run_all_examples_script_smoke(tmp_path: Path, monkeypatch):
    script = Path("scripts") / "run_all_examples.py"
    outdir = tmp_path / "gallery"
    monkeypatch.setattr(sys, "argv", [str(script), "--outdir", str(outdir)])
    runpy.run_path(str(script), run_name="__main__")

    assert (outdir / "random_walk_diffusion" / "particle_cloud.png").is_file()
    assert (outdir / "microchannel_biosensor" / "initial.png").is_file()
    assert (outdir / "spherical_cell_uptake" / "profiles.csv").is_file()
