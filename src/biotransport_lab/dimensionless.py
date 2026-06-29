"""Dimensionless numbers used in biological transport examples."""

from __future__ import annotations


def peclet_number(U_um_s: float, L_um: float, D_um2_s: float) -> float:
    """Return Pe = U L / D using um and seconds."""

    _check_positive(L_um=L_um, D_um2_s=D_um2_s)
    if U_um_s < 0:
        raise ValueError("U_um_s cannot be negative.")
    return U_um_s * L_um / D_um2_s


def reynolds_number(
    rho_kg_m3: float, U_um_s: float, L_um: float, mu_pa_s: float
) -> float:
    """Return Re = rho U L / mu after converting um/s and um to SI units."""

    _check_positive(rho_kg_m3=rho_kg_m3, L_um=L_um, mu_pa_s=mu_pa_s)
    if U_um_s < 0:
        raise ValueError("U_um_s cannot be negative.")
    return rho_kg_m3 * (U_um_s * 1e-6) * (L_um * 1e-6) / mu_pa_s


def damkohler_advection(k_s: float, L_um: float, U_um_s: float) -> float:
    """Return Da = k L / U for advection-reaction competition."""

    _check_nonnegative(k_s=k_s)
    _check_positive(L_um=L_um, U_um_s=U_um_s)
    return k_s * L_um / U_um_s


def damkohler_diffusion(k_s: float, L_um: float, D_um2_s: float) -> float:
    """Return Da = k L^2 / D for diffusion-reaction competition."""

    _check_nonnegative(k_s=k_s)
    _check_positive(L_um=L_um, D_um2_s=D_um2_s)
    return k_s * L_um * L_um / D_um2_s


def biot_like_number(surface_rate_um_s: float, L_um: float, D_um2_s: float) -> float:
    """Return Bi-like = surface reaction velocity times length divided by diffusion."""

    _check_nonnegative(surface_rate_um_s=surface_rate_um_s)
    _check_positive(L_um=L_um, D_um2_s=D_um2_s)
    return surface_rate_um_s * L_um / D_um2_s


def sherwood_number(mass_transfer_um_s: float, L_um: float, D_um2_s: float) -> float:
    """Return Sh = k_m L / D for a mass-transfer coefficient k_m."""

    return biot_like_number(mass_transfer_um_s, L_um, D_um2_s)


def interpret_value(name: str, value: float) -> str:
    """Return a short classroom interpretation for a dimensionless number."""

    if value < 0.1:
        regime = "small"
    elif value > 10:
        regime = "large"
    else:
        regime = "order-one"

    meanings = {
        "Pe": {
            "small": "diffusion dominates transport",
            "order-one": "advection and diffusion compete",
            "large": "advection dominates transport",
        },
        "Re": {
            "small": "viscous microflow behavior is expected",
            "order-one": "inertia may start to matter",
            "large": "inertia is important",
        },
        "Da_advection": {
            "small": "transport is faster than reaction",
            "order-one": "reaction and advection times are comparable",
            "large": "reaction is fast compared with advection",
        },
        "Da_diffusion": {
            "small": "diffusion is faster than reaction",
            "order-one": "reaction and diffusion times are comparable",
            "large": "reaction is fast compared with diffusion",
        },
        "Bi": {
            "small": "surface reaction is weak compared with diffusion",
            "order-one": "surface reaction and diffusion compete",
            "large": "surface uptake is strong enough to create depletion",
        },
        "Sh": {
            "small": "weak mass transfer relative to diffusion scaling",
            "order-one": "mass transfer follows the diffusion scale",
            "large": "enhanced mass transfer is expected",
        },
    }
    return meanings.get(name, {}).get(regime, f"{regime} value")


def summarize_dimensionless_numbers(
    *,
    U_um_s: float,
    L_um: float,
    D_um2_s: float,
    k_s: float = 0.0,
    rho_kg_m3: float = 1000.0,
    mu_pa_s: float = 1.0e-3,
    surface_rate_um_s: float = 0.0,
) -> dict[str, dict[str, float | str]]:
    """Return JSON-friendly dimensionless values and interpretations."""

    values = {
        "Pe": peclet_number(U_um_s, L_um, D_um2_s),
        "Re": reynolds_number(rho_kg_m3, U_um_s, L_um, mu_pa_s),
        "Da_advection": damkohler_advection(k_s, L_um, U_um_s) if U_um_s > 0 else 0.0,
        "Da_diffusion": damkohler_diffusion(k_s, L_um, D_um2_s),
        "Bi": biot_like_number(surface_rate_um_s, L_um, D_um2_s),
        "Sh": sherwood_number(surface_rate_um_s, L_um, D_um2_s),
    }
    return {
        name: {"value": value, "interpretation": interpret_value(name, value)}
        for name, value in values.items()
    }


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def _check_nonnegative(**values: float) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")
