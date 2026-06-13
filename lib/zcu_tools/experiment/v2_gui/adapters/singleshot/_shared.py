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

if TYPE_CHECKING:
    from zcu_tools.meta_tool.metadict import MetaDict

# MetaDict keys the ``singleshot/ge`` writeback owns (see GEAdapter writeback).
_G_CENTER_KEY = "g_center"
_E_CENTER_KEY = "e_center"
_GE_RADIUS_KEY = "ge_radius"

_MISSING_HINT = (
    "run the 'singleshot/ge' experiment first and apply its writeback "
    "(it proposes 'g_center' / 'e_center' / 'ge_radius' into the MetaDict)"
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


def _require_float(md: MetaDict, key: str) -> float:
    value = md.get(key)
    if value is None:
        raise RuntimeError(f"MetaDict is missing '{key}': {_MISSING_HINT}")
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(
        f"MetaDict '{key}' is not a numeric value (got {type(value).__name__}); "
        f"{_MISSING_HINT}"
    )
