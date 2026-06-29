"""Prescribed flow fields for biological transport teaching examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from biotransport_lab.core import CartesianDomain, FlowKind


def cartesian_grid(domain: CartesianDomain) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return x and y mesh grids in um with shape (ny, nx)."""

    x = np.linspace(0.0, domain.width_um, domain.nx)
    y = np.linspace(0.0, domain.height_um, domain.ny)
    return np.meshgrid(x, y)


def uniform_flow(
    domain: CartesianDomain, U_um_s: float, V_um_s: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Uniform flow in a rectangular biological assay channel."""

    x, _ = cartesian_grid(domain)
    return np.full_like(x, U_um_s, dtype=float), np.full_like(x, V_um_s, dtype=float)


def poiseuille_microchannel_flow(
    domain: CartesianDomain, mean_U_um_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Plane Poiseuille-like microchannel profile with the requested mean velocity."""

    x, y = cartesian_grid(domain)
    half_height = domain.height_um / 2.0
    eta = (y - half_height) / half_height
    u = 1.5 * mean_U_um_s * np.maximum(0.0, 1.0 - eta**2)
    return u, np.zeros_like(x)


def simple_shear_flow(
    domain: CartesianDomain, top_U_um_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Linear shear flow, useful for near-wall transport examples."""

    x, y = cartesian_grid(domain)
    u = top_U_um_s * y / domain.height_um
    return u, np.zeros_like(x)


def vortex_flow(
    domain: CartesianDomain, edge_U_um_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simple recirculating field for mixing and trapped-plume demonstrations."""

    x, y = cartesian_grid(domain)
    x0 = domain.width_um / 2.0
    y0 = domain.height_um / 2.0
    length = min(domain.width_um, domain.height_um) / 2.0
    omega = edge_U_um_s / max(length, 1e-12)
    u = -omega * (y - y0)
    v = omega * (x - x0)
    speed = np.hypot(u, v).max()
    if speed > 0:
        u *= edge_U_um_s / speed
        v *= edge_U_um_s / speed
    return u, v


def radial_flow(
    domain: CartesianDomain,
    speed_um_s: float,
    center_x_um: float | None = None,
    center_y_um: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Radial inward or outward flow from a source, sink, vessel, or bead."""

    x, y = cartesian_grid(domain)
    cx = domain.width_um / 2.0 if center_x_um is None else center_x_um
    cy = domain.height_um / 2.0 if center_y_um is None else center_y_um
    dx = x - cx
    dy = y - cy
    radius = np.hypot(dx, dy)
    safe_radius = np.where(radius == 0.0, 1.0, radius)
    u = speed_um_s * dx / safe_radius
    v = speed_um_s * dy / safe_radius
    u[radius == 0.0] = 0.0
    v[radius == 0.0] = 0.0
    return u, v


def prescribed_flow(
    kind: FlowKind, domain: CartesianDomain, U_um_s: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a named prescribed velocity field."""

    if kind == "uniform":
        return uniform_flow(domain, U_um_s)
    if kind == "poiseuille":
        return poiseuille_microchannel_flow(domain, U_um_s)
    if kind == "shear":
        return simple_shear_flow(domain, U_um_s)
    if kind == "vortex":
        return vortex_flow(domain, U_um_s)
    if kind == "radial":
        return radial_flow(domain, U_um_s)
    raise ValueError(f"Unknown flow kind: {kind}")
