"""Optional OpenFOAM adapter.

OpenFOAM is treated as an external high-fidelity backend. The Python teaching
workflow must work even when no OpenFOAM command is installed.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenFOAMStatus:
    """OpenFOAM availability report."""

    installed: bool
    version: str | None
    foam_version_path: str | None
    message: str

    def to_dict(self) -> dict[str, bool | str | None]:
        return {
            "installed": self.installed,
            "version": self.version,
            "foamVersion_path": self.foam_version_path,
            "message": self.message,
        }


def get_openfoam_status() -> dict[str, bool | str | None]:
    """Detect whether the external foamVersion command is available."""

    foam_version_path = shutil.which("foamVersion")
    if foam_version_path is None:
        return OpenFOAMStatus(
            installed=False,
            version=None,
            foam_version_path=None,
            message="OpenFOAM command foamVersion was not found. Python solvers remain available.",
        ).to_dict()

    try:
        completed = subprocess.run(
            ["foamVersion"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return OpenFOAMStatus(
            installed=False,
            version=None,
            foam_version_path=foam_version_path,
            message=f"foamVersion was found but could not run: {exc}",
        ).to_dict()

    output = (completed.stdout or completed.stderr).strip()
    installed = completed.returncode == 0
    return OpenFOAMStatus(
        installed=installed,
        version=output if installed else None,
        foam_version_path=foam_version_path,
        message="OpenFOAM is available." if installed else "foamVersion returned an error.",
    ).to_dict()


def create_scalar_transport_case(case_dir: str | Path) -> dict[str, str]:
    """Create a minimal placeholder case directory for optional exercises."""

    case_path = Path(case_dir)
    (case_path / "0").mkdir(parents=True, exist_ok=True)
    (case_path / "constant").mkdir(parents=True, exist_ok=True)
    (case_path / "system").mkdir(parents=True, exist_ok=True)
    (case_path / "README.txt").write_text(
        "Placeholder OpenFOAM scalar-transport case.\n"
        "Use this directory as a starting point after installing OpenFOAM externally.\n",
        encoding="utf-8",
    )
    return {"case_dir": str(case_path), "message": "Created optional case skeleton."}


def run_openfoam_case(case_dir: str | Path | None = None) -> dict[str, object]:
    """Run an optional OpenFOAM case only when OpenFOAM is available."""

    status = get_openfoam_status()
    if not status["installed"]:
        return {
            "started": False,
            "status": status,
            "message": "OpenFOAM backend is disabled because OpenFOAM is unavailable.",
        }

    case_path = Path("openfoam/cases/demo" if case_dir is None else case_dir)
    create_scalar_transport_case(case_path)
    return {
        "started": False,
        "status": status,
        "case_dir": str(case_path),
        "message": "Case skeleton created. Automatic solver execution is left to the instructor.",
    }
