"""Translate low-level exceptions into friendly, actionable UI messages.

Pure functions (no Qt), shared by the panels that surface errors. Each maps a
common dispersive-pipeline failure to a clear "what went wrong + how to fix it"
message, keeping the raw error on a ``Details:`` line (the full traceback is in the
debug log).
"""

from __future__ import annotations

import os
from typing import Union


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

    return f"{head}\n\nDetails: {raw if raw else type(exc).__name__}"


def friendly_fit_message(action: str, exc: Union[Exception, str]) -> str:
    """Message for a preprocess / auto-fit / export failure.

    ``action`` is "Preprocess" / "Auto-fit" / "Export". ``exc`` may be the exception
    (caught locally) or its message string (crossing a worker signal). Covers the
    domain preconditions and delegates a params.json write failure to
    ``friendly_io_message``.
    """
    raw = (str(exc) if not isinstance(exc, str) else exc).strip()
    low = raw.lower()

    if "unable to open file" in low or ("unable to" in low and "file" in low):
        wrapped = exc if isinstance(exc, Exception) else OSError(raw)
        return friendly_io_message("Export", "params.json", wrapped)

    if "no fluxonium fit inputs" in low or "load params.json first" in low:
        head = "Load the fit inputs (params.json) first."
    elif "no one-tone" in low:
        head = "Load a one-tone spectrum first."
    elif "no preprocessing" in low:
        head = "Run preprocessing before fitting g."
    elif "no bare_rf" in low:
        head = "No bare_rf set — load the fit inputs to seed it."
    elif "no dispersive fit result" in low:
        head = "Nothing to export yet — fit g first."
    else:
        head = f"{action} failed."

    fallback = type(exc).__name__ if isinstance(exc, Exception) else "error"
    return f"{head}\n\nDetails: {raw if raw else fallback}"
