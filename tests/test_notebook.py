from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path("notebooks") / "BioTransport_CFD_Lab.ipynb"


def test_notebook_structure():
    nbformat = pytest.importorskip("nbformat")
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    markdown = "\n".join(
        "".join(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "Random walk to diffusion" in markdown
    assert "Cylindrical radial transport" in markdown
    assert "Spherical radial transport" in markdown
    assert "Limitations" in markdown


def test_notebook_contains_inline_visualizations():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])
    assert "show_saved_figures" in source
    assert "display(Image" in source
    assert "dimensionless_numbers.png" in source


def test_notebook_executes_with_nbclient(tmp_path: Path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=180, kernel_name="python3")
    client.execute(resources={"metadata": {"path": str(Path.cwd())}})
    assert (Path("outputs") / "notebook" / "microchannel_biosensor" / "final.png").exists()
