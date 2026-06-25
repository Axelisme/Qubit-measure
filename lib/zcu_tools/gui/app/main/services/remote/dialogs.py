"""Dialog naming + factory contract for RemoteControlAdapter.

``DialogName`` is the wire enum a remote caller uses with
``dialog.open`` / ``dialog.close`` / ``dialog.list_open``. ``MainWindow``
owns the registry (``self._open_dialogs``) and the per-name factory that
builds or reuses a ``QDialog`` when the remote (or a UI click) asks to open
one.

All dialogs are opened **non-modal** (``dlg.open()``) so that the Qt event
loop keeps pumping while the dialog is visible — this is mandatory for
remote-driven flows where a follow-up RPC must still be dispatchable.
Most dialogs use ``WA_DeleteOnClose`` + ``finished`` cleanup; expensive
persistent dialogs can instead hide on close and stay cached in the registry.

The ``STARTUP`` factory is registered late by ``gui/app.py``
because the startup dialog construction needs ``startup_mode=True`` and is
typically opened by the application bootstrap, not by an explicit click.
"""

from __future__ import annotations

from enum import Enum


class DialogName(str, Enum):
    """Wire-stable identifiers for remotely controllable dialogs."""

    SETUP = "setup"
    DEVICE = "device"
    PREDICTOR = "predictor"
    INSPECT = "inspect"
    ARB_WAVEFORM = "arb_waveform"
    STARTUP = "startup"


def parse_dialog_name(value: object) -> DialogName:
    """Coerce a wire string into a ``DialogName`` enum.

    Accepts the lowercase wire form (``"setup"``) and the upper-case enum
    name (``"SETUP"``) for client ergonomics. Raises ``ValueError`` if the
    name is unknown — callers translate that into ``invalid_params``.
    """
    if isinstance(value, DialogName):
        return value
    if not isinstance(value, str):
        raise ValueError(f"dialog name must be a string, got {type(value).__name__}")
    lowered = value.lower()
    for name in DialogName:
        if name.value == lowered:
            return name
    upper = value.upper()
    for name in DialogName:
        if name.name == upper:
            return name
    raise ValueError(f"unknown dialog name: {value!r}")


__all__ = ["DialogName", "parse_dialog_name"]
