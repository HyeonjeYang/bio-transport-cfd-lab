"""Optional OpenFOAM adapter for a small scalar-transport teaching case.

OpenFOAM is treated as an external high-fidelity backend. The Python teaching
workflow must work even when no OpenFOAM command is installed.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OpenFOAMStatus:
    """OpenFOAM availability report."""

    installed: bool
    version: str | None
    foam_version_path: str | None
    block_mesh_path: str | None
    scalar_transport_path: str | None
    message: str

    def to_dict(self) -> dict[str, bool | str | None]:
        return {
            "installed": self.installed,
            "version": self.version,
            "foamVersion_path": self.foam_version_path,
            "blockMesh_path": self.block_mesh_path,
            "scalarTransportFoam_path": self.scalar_transport_path,
            "message": self.message,
        }


@dataclass(frozen=True)
class OpenFOAMCaseConfig:
    """Parameters for the optional OpenFOAM scalar-transport case."""

    D_um2_s: float = 80.0
    U_um_s: float = 200.0
    length_um: float = 220.0
    height_um: float = 90.0
    nx: int = 44
    ny: int = 18
    total_time_s: float = 0.2
    write_interval_s: float = 0.1
    inlet_concentration: float = 1.0

    def __post_init__(self) -> None:
        if self.D_um2_s <= 0 or self.length_um <= 0 or self.height_um <= 0:
            raise ValueError("D, length, and height must be positive.")
        if self.U_um_s < 0 or self.inlet_concentration < 0:
            raise ValueError("Velocity and inlet concentration cannot be negative.")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("OpenFOAM grid needs at least two cells in x and y.")
        if self.total_time_s <= 0 or self.write_interval_s <= 0:
            raise ValueError("OpenFOAM times must be positive.")

    @classmethod
    def from_web_params(cls, params: dict[str, Any] | None) -> OpenFOAMCaseConfig:
        """Build a case config from the shared web request parameters."""

        params = {} if params is None else params
        total_time = min(float(params.get("total_time", 0.2)), 1.0)
        return cls(
            D_um2_s=float(params.get("D", 80.0)),
            U_um_s=float(params.get("U", 200.0)),
            total_time_s=max(total_time, 0.02),
            write_interval_s=max(total_time / 2.0, 0.01),
        )


def get_openfoam_status() -> dict[str, bool | str | None]:
    """Detect whether external OpenFOAM commands are available."""

    foam_version_path = shutil.which("foamVersion")
    block_mesh_path = shutil.which("blockMesh")
    scalar_transport_path = shutil.which("scalarTransportFoam")
    if foam_version_path is None:
        return OpenFOAMStatus(
            installed=False,
            version=None,
            foam_version_path=None,
            block_mesh_path=block_mesh_path,
            scalar_transport_path=scalar_transport_path,
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
            block_mesh_path=block_mesh_path,
            scalar_transport_path=scalar_transport_path,
            message=f"foamVersion was found but could not run: {exc}",
        ).to_dict()

    output = (completed.stdout or completed.stderr).strip()
    commands_ready = completed.returncode == 0 and block_mesh_path and scalar_transport_path
    if completed.returncode == 0 and not commands_ready:
        message = "OpenFOAM was found, but blockMesh or scalarTransportFoam is missing."
    else:
        message = "OpenFOAM is available." if commands_ready else "foamVersion returned an error."
    return OpenFOAMStatus(
        installed=bool(commands_ready),
        version=output if completed.returncode == 0 else None,
        foam_version_path=foam_version_path,
        block_mesh_path=block_mesh_path,
        scalar_transport_path=scalar_transport_path,
        message=message,
    ).to_dict()


def create_scalar_transport_case(
    case_dir: str | Path, config: OpenFOAMCaseConfig | None = None
) -> dict[str, object]:
    """Create a runnable OpenFOAM scalarTransportFoam microchannel case."""

    config = OpenFOAMCaseConfig() if config is None else config
    case_path = Path(case_dir).resolve()
    for folder in ["0", "constant", "system", "postProcessing"]:
        (case_path / folder).mkdir(parents=True, exist_ok=True)

    (case_path / "system" / "blockMeshDict").write_text(_block_mesh_dict(config), encoding="utf-8")
    (case_path / "system" / "controlDict").write_text(_control_dict(config), encoding="utf-8")
    (case_path / "system" / "fvSchemes").write_text(_fv_schemes(), encoding="utf-8")
    (case_path / "system" / "fvSolution").write_text(_fv_solution(), encoding="utf-8")
    (case_path / "constant" / "transportProperties").write_text(
        _transport_properties(config), encoding="utf-8"
    )
    (case_path / "0" / "U").write_text(_u_field(config), encoding="utf-8")
    (case_path / "0" / "T").write_text(_t_field(config), encoding="utf-8")
    (case_path / "README.txt").write_text(
        "OpenFOAM scalarTransportFoam microchannel case.\n"
        "Units are SI inside OpenFOAM; input parameters are converted from um and seconds.\n",
        encoding="utf-8",
    )
    return {
        "case_dir": str(case_path),
        "config": asdict(config),
        "message": "Created runnable scalarTransportFoam microchannel case.",
    }


def run_openfoam_case(
    case_dir: str | Path | None = None,
    config: OpenFOAMCaseConfig | None = None,
    timeout_s: int = 90,
) -> dict[str, object]:
    """Create and run the optional OpenFOAM case when OpenFOAM is installed."""

    status = get_openfoam_status()
    if not status["installed"]:
        return {
            "started": False,
            "completed": False,
            "status": status,
            "message": "OpenFOAM backend is disabled because required commands are unavailable.",
        }

    config = OpenFOAMCaseConfig() if config is None else config
    case_path = Path("openfoam/cases/demo" if case_dir is None else case_dir).resolve()
    create_scalar_transport_case(case_path, config)

    block_mesh = _run_command(["blockMesh", "-case", str(case_path)], case_path, timeout_s)
    if block_mesh["returncode"] != 0:
        return {
            "started": True,
            "completed": False,
            "status": status,
            "case_dir": str(case_path),
            "commands": [block_mesh],
            "message": "blockMesh failed. See log.blockMesh.txt in the case directory.",
        }

    solver = _run_command(["scalarTransportFoam", "-case", str(case_path)], case_path, timeout_s)
    csv_path = export_latest_scalar_csv(case_path, config)
    return {
        "started": True,
        "completed": solver["returncode"] == 0 and csv_path is not None,
        "status": status,
        "case_dir": str(case_path),
        "commands": [block_mesh, solver],
        "result_csv": None if csv_path is None else str(csv_path),
        "message": (
            "OpenFOAM run completed."
            if solver["returncode"] == 0
            else "scalarTransportFoam failed. See log.scalarTransportFoam.txt in the case directory."
        ),
    }


def export_latest_scalar_csv(
    case_dir: str | Path, config: OpenFOAMCaseConfig | None = None
) -> Path | None:
    """Export the latest OpenFOAM scalar field T to a simple CSV file."""

    config = OpenFOAMCaseConfig() if config is None else config
    case_path = Path(case_dir)
    latest = _latest_time_dir(case_path)
    if latest is None:
        return None
    field_path = latest / "T"
    if not field_path.exists():
        return None
    values = _read_internal_scalar_field(field_path, config.nx * config.ny)
    if values is None:
        return None

    output = case_path / "postProcessing" / "latest_T.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    dx = config.length_um / config.nx
    dy = config.height_um / config.ny
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_um", "y_um", "concentration"])
        for j in range(config.ny):
            for i in range(config.nx):
                index = j * config.nx + i
                writer.writerow([(i + 0.5) * dx, (j + 0.5) * dy, values[index]])
    return output


def _run_command(command: list[str], case_path: Path, timeout_s: int) -> dict[str, object]:
    name = command[0]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        stdout = ""
        stderr = str(exc)
        returncode = 124

    log_path = case_path / f"log.{name}.txt"
    log_path.write_text(stdout + "\n--- stderr ---\n" + stderr, encoding="utf-8")
    return {"command": " ".join(command), "returncode": returncode, "log": str(log_path)}


def _latest_time_dir(case_path: Path) -> Path | None:
    time_dirs: list[tuple[float, Path]] = []
    for child in case_path.iterdir():
        if not child.is_dir():
            continue
        try:
            value = float(child.name)
        except ValueError:
            continue
        if value > 0:
            time_dirs.append((value, child))
    if not time_dirs:
        return None
    return sorted(time_dirs, key=lambda item: item[0])[-1][1]


def _read_internal_scalar_field(field_path: Path, expected_cells: int) -> list[float] | None:
    text = field_path.read_text(encoding="utf-8", errors="ignore")
    uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)\s*;", text)
    if uniform:
        return [float(uniform.group(1))] * expected_cells

    match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )
    if match is None:
        return None
    count = int(match.group(1))
    raw_values = [float(item) for item in match.group(2).split()]
    if count != expected_cells or len(raw_values) != expected_cells:
        return None
    return raw_values


def _foam_header(class_name: str, object_name: str) -> str:
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def _block_mesh_dict(config: OpenFOAMCaseConfig) -> str:
    length = config.length_um
    height = config.height_um
    return (
        _foam_header("dictionary", "blockMeshDict")
        + f"""
convertToMeters 1e-6;

vertices
(
    (0 0 0)
    ({length:g} 0 0)
    ({length:g} {height:g} 0)
    (0 {height:g} 0)
    (0 0 1)
    ({length:g} 0 1)
    ({length:g} {height:g} 1)
    (0 {height:g} 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({config.nx} {config.ny} 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    lowerWall
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    upperWall
    {{
        type wall;
        faces ((3 7 6 2));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);

mergePatchPairs ();
"""
    )


def _control_dict(config: OpenFOAMCaseConfig) -> str:
    return (
        _foam_header("dictionary", "controlDict")
        + f"""
application     scalarTransportFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {config.total_time_s:g};
deltaT          {min(config.write_interval_s / 10.0, 0.005):g};
writeControl    runTime;
writeInterval   {config.write_interval_s:g};
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    )


def _fv_schemes() -> str:
    return (
        _foam_header("dictionary", "fvSchemes")
        + """
ddtSchemes
{
    default Euler;
}

gradSchemes
{
    default Gauss linear;
}

divSchemes
{
    default none;
    div(phi,T) Gauss upwind;
}

laplacianSchemes
{
    default Gauss linear corrected;
}

interpolationSchemes
{
    default linear;
}

snGradSchemes
{
    default corrected;
}
"""
    )


def _fv_solution() -> str:
    return (
        _foam_header("dictionary", "fvSolution")
        + """
solvers
{
    T
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0;
    }
}
"""
    )


def _transport_properties(config: OpenFOAMCaseConfig) -> str:
    diffusivity_m2_s = config.D_um2_s * 1e-12
    return (
        _foam_header("dictionary", "transportProperties")
        + f"""
DT              DT [0 2 -1 0 0 0 0] {diffusivity_m2_s:.8e};
"""
    )


def _u_field(config: OpenFOAMCaseConfig) -> str:
    velocity_m_s = config.U_um_s * 1e-6
    return (
        _foam_header("volVectorField", "U")
        + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({velocity_m_s:.8e} 0 0);

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform ({velocity_m_s:.8e} 0 0);
    }}
    outlet
    {{
        type zeroGradient;
    }}
    lowerWall
    {{
        type noSlip;
    }}
    upperWall
    {{
        type noSlip;
    }}
    frontAndBack
    {{
        type empty;
    }}
}}
"""
    )


def _t_field(config: OpenFOAMCaseConfig) -> str:
    return (
        _foam_header("volScalarField", "T")
        + f"""
dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform {config.inlet_concentration:g};
    }}
    outlet
    {{
        type zeroGradient;
    }}
    lowerWall
    {{
        type zeroGradient;
    }}
    upperWall
    {{
        type zeroGradient;
    }}
    frontAndBack
    {{
        type empty;
    }}
}}
"""
    )
