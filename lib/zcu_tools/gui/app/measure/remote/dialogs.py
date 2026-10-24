"""Dialog naming contract shared by the measure window and its registry.

``DialogName`` identifies dialogs opened by the measure GUI
and the key ``MainWindow.open_dialog`` uses for its named-dialog registry.
``MainDialogRegistry`` owns the per-name factory that builds or reuses a
``QDialog`` for a toolbar click or the app's ``after_show`` hook.

All dialogs are opened **non-modal** (``dlg.open()``) so that the Qt event
loop keeps pumping while the dialog is visible — this is mandatory for
remote-driven flows where a follow-up RPC must still be dispatchable.
Most dialogs use ``WA_DeleteOnClose`` + ``finished`` cleanup; expensive
persistent dialogs can instead hide on close and stay cached in the registry.
"""

from __future__ import annotations

from enum import Enum


class DialogName(str, Enum):
    """Stable identifiers for the measure window's named dialogs."""

    SETUP = "setup"
    SOC = "soc"
    DEVICE = "device"
    PREDICTOR = "predictor"
    INSPECT = "inspect"
    ARB_WAVEFORM = "arb_waveform"


__all__ = ["DialogName"]
