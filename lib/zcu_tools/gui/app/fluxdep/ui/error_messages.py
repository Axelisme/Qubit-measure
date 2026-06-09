"""Translate low-level exceptions into friendly, actionable UI messages.

Pure functions (no Qt), shared by the windows/panels that surface errors. Each
maps the common ways an action fails to a clear "what went wrong + how to fix it"
message, and keeps the raw error on a ``Details:`` line for the curious (the full
traceback is already in the debug log). The message *skeleton* (rule ladder +
``Details:`` tail + the fit→IO redirect) is the shared ``gui.error_messages``
framework; the fluxdep-specific domain rules live here.
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
    """Message for a file-IO failure (load / restore / export).

    ``action`` is the verb ("Load" / "Restore" / "Export") so the wording fits;
    ``filepath`` is the file involved.
    """
    raw = str(exc).strip()
    name = os.path.basename(filepath) or filepath

    if isinstance(exc, FileNotFoundError) or "unable to open file" in raw.lower():
        if action == "Export":
            head = (
                f"Could not write {name!r}. The folder may not exist or is not "
                "writable — pick a different location."
            )
        elif not os.path.exists(filepath):
            head = f"File not found: {name!r}. Check the path and try again."
        else:
            head = f"Could not open {name!r} — it may not be a valid HDF5 file."
    elif "file signature not found" in raw.lower() or "not a valid" in raw.lower():
        head = f"{name!r} is not a valid HDF5 file."
    elif action == "Restore" and (
        isinstance(exc, (AssertionError, KeyError)) or not raw
    ):
        head = (
            f"{name!r} is not a processed spectrums.hdf5 (it lacks the expected "
            "structure). To load a raw spectrum, use Add instead of Restore."
        )
    elif "no frequency axis" in raw.lower():
        head = (
            f"{name!r} is not a 2D spectrum (no frequency axis). Add expects a "
            "device-value × frequency sweep."
        )
    elif "no spectra to export" in raw.lower():
        head = "Nothing to export — load and annotate a spectrum first."
    else:
        head = f"{action} of {name!r} failed."

    return head + details_tail(raw, type(exc).__name__)


# Domain rules for friendly_fit_message — first match wins (substring on the
# lowercased raw error). Covers the domain preconditions (no database / no points /
# no result / no aligned spectrum), the mirror-needs-sample_f transition error and
# the no-candidate search outcome.
_FIT_RULES: list[FriendlyRule] = [
    (lambda low: "no database path" in low, "Select a database file before searching."),
    (
        lambda low: "no selected points" in low,
        "No points to fit — select points on a spectrum first.",
    ),
    (
        lambda low: "sample_f is required for mirror" in low,
        "The transitions include a 'mirror' category, which needs a non-zero "
        "sample_f. Set sample_f, or remove the mirror transitions.",
    ),
    (
        lambda low: "no valid candidate" in low or "infeasible" in low,
        "No match found in the database for these bounds. Widen the EJ / EC / "
        "EL bounds (or pick a different preset) and search again.",
    ),
    (lambda low: "no fit result" in low, "Nothing to export yet — run a search first."),
    (
        lambda low: "no aligned spectrum" in low,
        "Align at least one spectrum before exporting params.json.",
    ),
    (
        lambda low: "no result_dir" in low,
        "No output folder set — choose a save path for params.json.",
    ),
]


def friendly_fit_message(action: str, exc: Union[Exception, str]) -> str:
    """Message for a database-search / params-export failure.

    ``action`` is "Search" or "Export". ``exc`` may be the exception (export,
    caught locally) or just its message string (search, which crosses a worker
    signal as a string). A params.json write failure is delegated to
    ``friendly_io_message``; otherwise the domain rules above match.
    """
    raw = normalize_raw(exc)
    redirect = fit_io_redirect(exc, raw, friendly_io_message)
    if redirect is not None:
        return redirect
    fallback = type(exc).__name__ if isinstance(exc, Exception) else "error"
    return friendly_from_rules(action, raw, _FIT_RULES, fallback)
