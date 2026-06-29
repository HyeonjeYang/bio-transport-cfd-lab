from __future__ import annotations

from pathlib import Path

import pytest


def test_notebook_structure():
    nbformat = pytest.importorskip("nbformat")
    notebook_path = Path("notebooks") / "BioTransport_CFD_Lab.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    markdown = "\n".join(
        "".join(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "Random walk to diffusion" in markdown
    assert "Cylindrical radial transport" in markdown
    assert "Spherical radial transport" in markdown
    assert "Limitations" in markdown


def test_notebook_executes_with_nbclient(tmp_path: Path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    notebook_path = Path("notebooks") / "BioTransport_CFD_Lab.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=180, kernel_name="python3")
    client.execute(resources={"metadata": {"path": str(Path.cwd())}})
    assert (Path("outputs") / "notebook" / "microchannel_biosensor" / "final.png").exists()
