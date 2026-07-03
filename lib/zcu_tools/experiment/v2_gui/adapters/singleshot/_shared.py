"""Shared helpers for the single-shot downstream adapters.

The single-shot sweep / check experiments classify each raw shot inside the
program against the |g>/|e> IQ-cluster centres + radius produced by the
``singleshot/ge`` experiment (its writeback proposes ``g_center`` / ``e_center``
/ ``ge_radius`` into the MetaDict). Their domain ``run`` (mist) or ``analyze``
(check) therefore needs those three values as explicit inputs.

``read_ge_centers`` is the single read point: it pulls the trio off a MetaDict
and fast-fails (pointing the user back to ``singleshot/ge``) if any is missing,
so a downstream run never silently classifies against absent/garbage centres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.experiment.v2_gui.adapters.shared.ctx_helpers import md_has_key
from zcu_tools.gui.app.main.adapter import (
    DirectValue,
    EvalValue,
    ScalarValue,
    SweepValue,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.meta_tool.metadict import MetaDict

# MetaDict keys the ``singleshot/ge`` writeback owns (see GEAdapter writeback).
_G_CENTER_KEY = "g_center"
_E_CENTER_KEY = "e_center"
_GE_RADIUS_KEY = "ge_radius"

_MISSING_HINT = (
    "run the 'singleshot/ge' experiment first and apply its writeback "
    "(it proposes 'g_center' / 'e_center' / 'ge_radius' into the MetaDict)"
)

# MetaDict keys the dispersive-shift experiment owns (chi + resonator linewidth).
_CHI_KEY = "chi"
_KAPPA_KEY = "rf_w"

_CHI_KAPPA_HINT = (
    "run the dispersive-shift experiment first so 'chi' (dispersive shift) and "
    "'rf_w' (resonator linewidth kappa) are present in the MetaDict"
)


def read_ge_centers(md: MetaDict) -> tuple[complex, complex, float]:
    """Read the ``(g_center, e_center, radius)`` GE classification trio from md.

    ``g_center`` / ``e_center`` are complex IQ-cluster centres (kept complex —
    the GE writeback round-trips complex md values end-to-end); ``radius`` is the
    optimised classification radius (float). Fast-fails with an actionable message
    if any of the three is absent or ``None`` — a downstream single-shot run must
    never classify against missing centres.
    """
    g_center = _require_complex(md, _G_CENTER_KEY)
    e_center = _require_complex(md, _E_CENTER_KEY)
    radius = _require_float(md, _GE_RADIUS_KEY)
    return g_center, e_center, radius


def read_chi_kappa(md: MetaDict) -> tuple[float, float]:
    """Read the ``(chi, kappa)`` AC-Stark fit inputs from md.

    ``chi`` is the dispersive shift (md 'chi', MHz) and ``kappa`` is the
    resonator linewidth (md 'rf_w', MHz) — both required by the AC-Stark
    coefficient fit, which derives ``eta = kappa² / (kappa² + chi²)`` internally
    (matching the twotone AcStarkExp.analyze). Fast-fails with an actionable
    message if either is absent, so the fit never runs against missing
    calibration.
    """
    chi = _require_float(md, _CHI_KEY, _CHI_KAPPA_HINT)
    kappa = _require_float(md, _KAPPA_KEY, _CHI_KAPPA_HINT)
    return chi, kappa


def readout_probe_freq(ctx: ExpContext) -> ScalarValue:
    """Default a probe tone to the readout/resonator frequency calibration."""
    if md_has_key(ctx, "readout_f"):
        return EvalValue(expr="readout_f")
    if md_has_key(ctx, "r_f"):
        return EvalValue(expr="r_f")
    return DirectValue(6000.0)


def readout_probe_freq_range(
    ctx: ExpContext, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Default a probe frequency sweep around readout_f, falling back to r_f."""
    center_key = "readout_f" if md_has_key(ctx, "readout_f") else "r_f"
    center_raw = ctx.md.get(center_key)
    center = float(center_raw) if isinstance(center_raw, (int, float)) else 6000.0
    width_raw = ctx.md.get("rf_w")
    width = float(width_raw) if isinstance(width_raw, (int, float)) else 500.0
    half_span = span_factor * width if width > 0 else 30.0

    width_term = "rf_w" if span_factor == 1.0 else f"{span_factor} * rf_w"
    if md_has_key(ctx, center_key) and md_has_key(ctx, "rf_w"):
        return SweepValue(
            start=EvalValue(expr=f"{center_key} - {width_term}"),
            stop=EvalValue(expr=f"{center_key} + {width_term}"),
            expts=expts,
        )
    if md_has_key(ctx, center_key):
        return SweepValue(
            start=EvalValue(expr=f"{center_key} - {half_span}"),
            stop=EvalValue(expr=f"{center_key} + {half_span}"),
            expts=expts,
        )
    return SweepValue(start=center - half_span, stop=center + half_span, expts=expts)


def _require_complex(md: MetaDict, key: str) -> complex:
    value = md.get(key)
    if value is None:
        raise RuntimeError(f"MetaDict is missing '{key}': {_MISSING_HINT}")
    # Accept the in-process complex object as-is; coerce int/float/str (the
    # MetaDict persistence form) so a reloaded md still yields complex.
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float, str)):
        return complex(value)
    raise RuntimeError(
        f"MetaDict '{key}' is not a complex value (got {type(value).__name__}); "
        f"{_MISSING_HINT}"
    )


def _require_float(md: MetaDict, key: str, hint: str = _MISSING_HINT) -> float:
    value = md.get(key)
    if value is None:
        raise RuntimeError(f"MetaDict is missing '{key}': {hint}")
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(
        f"MetaDict '{key}' is not a numeric value (got {type(value).__name__}); {hint}"
    )
