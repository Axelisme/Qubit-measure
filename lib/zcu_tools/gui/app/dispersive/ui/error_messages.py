"""Translate low-level exceptions into friendly, actionable UI messages.

Pure functions (no Qt), shared by the panels that surface errors. Each maps a
common dispersive-pipeline failure to a clear "what went wrong + how to fix it"
message, keeping the raw error on a ``Details:`` line (the full traceback is in the
debug log). The message *skeleton* (rule ladder + ``Details:`` tail + the fit→IO
redirect) is the shared ``gui.error_messages`` framework; the dispersive-specific
domain rules live here.
"""

from __future__ import annotations

import os
from typing import Union

from zcu_tools.gui.error_messages import (
    FriendlyRule,
    details_tail,
    fit_io_redirect,
    friendly_from_rules,
    normalize_raw,
)


def friendly_io_message(action: str, filepath: str, exc: Exception) -> str:
    """Message for a file-IO failure (load inputs / load onetone / export).

    ``action`` is the verb ("Load" / "Export"); ``filepath`` is the file involved.
    """
    raw = str(exc).strip()
    name = os.path.basename(filepath) or filepath

    if isinstance(exc, FileNotFoundError) or "unable to open file" in raw.lower():
        if action == "Export":
            head = (
                f"Could not write {name!r}. The params.json must already exist (it "
                "holds the fluxdep_fit section) — load the fit inputs first."
            )
        elif not os.path.exists(filepath):
            head = f"File not found: {name!r}. Check the path and try again."
        else:
            head = f"Could not open {name!r} — it may not be a valid file."
    elif "file signature not found" in raw.lower() or "not a valid" in raw.lower():
        head = f"{name!r} is not a valid HDF5 file."
    elif "no frequency axis" in raw.lower():
        head = (
            f"{name!r} is not a 2D spectrum (no frequency axis). A one-tone is a "
            "device-value × frequency sweep."
        )
    elif "no 'fluxdep_fit'" in raw.lower() or "run fluxdep-gui first" in raw.lower():
        head = (
            "This params.json has no fluxdep_fit section. Run fluxdep-gui first to "
            "fit (EJ, EC, EL) and the flux alignment dispersive needs as inputs."
        )
    else:
        head = f"{action} of {name!r} failed."

    return head + details_tail(raw, type(exc).__name__)


# Domain rules for friendly_fit_message — first match wins (substring on the
# lowercased raw error). Covers the dispersive-pipeline preconditions.
_FIT_RULES: list[FriendlyRule] = [
    (
        lambda low: "no fluxonium fit inputs" in low or "load params.json first" in low,
        "Load the fit inputs (params.json) first.",
    ),
    (lambda low: "no one-tone" in low, "Load a one-tone spectrum first."),
    (lambda low: "no preprocessing" in low, "Run preprocessing before fitting g."),
    (
        lambda low: "no bare_rf" in low,
        "No bare_rf set — load the fit inputs to seed it.",
    ),
    (
        lambda low: "no dispersive fit result" in low,
        "Nothing to export yet — fit g first.",
    ),
]


def friendly_fit_message(action: str, exc: Union[Exception, str]) -> str:
    """Message for a preprocess / auto-fit / export failure.

    ``action`` is "Preprocess" / "Auto-fit" / "Export". ``exc`` may be the exception
    (caught locally) or its message string (crossing a worker signal). A params.json
    write failure is delegated to ``friendly_io_message``; otherwise the domain rules
    above match.
    """
    raw = normalize_raw(exc)
    redirect = fit_io_redirect(exc, raw, friendly_io_message)
    if redirect is not None:
        return redirect
    fallback = type(exc).__name__ if isinstance(exc, Exception) else "error"
    return friendly_from_rules(action, raw, _FIT_RULES, fallback)
