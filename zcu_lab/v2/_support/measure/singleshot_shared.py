"""Shared defaults and analyze-time helpers for single-shot adapters.

Sweep runs classify shots using the GE centre/radius trio frozen from resolved
cfg, not the live MetaDict. ``singleshot/ge`` writeback can seed those defaults,
but direct cfg values work too. ``singleshot/check`` instead reads the trio from
MetaDict during analyze; ``read_ge_centers`` fast-fails there if any is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zcu_tools.gui.cfg import (
    EvalValue,
    ScalarValue,
    SweepValue,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.resources.context.metadict import MetaDict

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
    if any of the three is absent or ``None`` during ``singleshot/check`` analyze.
    Sweep runs consume their resolved cfg values and do not call this helper.
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


def readout_probe_freq(ctx: SessionEnv) -> ScalarValue:
    """Store $resonator.freq without reading ctx; missing calibration is Invalid."""
    del ctx  # Defaults do not read the source before resource resolution.
    return EvalValue(expr="$resonator.freq")


def readout_probe_freq_range(
    ctx: SessionEnv, expts: int, *, span_factor: float = 1.5
) -> SweepValue:
    """Store a live $resonator.freq ± span_factor*$resonator.kappa sweep.

    ctx is not read; missing aliases/calibration become Invalid at resolution.
    """
    del ctx  # Defaults do not read the source before resource resolution.
    return SweepValue(
        start=EvalValue(f"$resonator.freq - {span_factor} * $resonator.kappa"),
        stop=EvalValue(f"$resonator.freq + {span_factor} * $resonator.kappa"),
        expts=expts,
    )


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
