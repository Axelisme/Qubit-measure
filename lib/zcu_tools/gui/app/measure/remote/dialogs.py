"""Dialog naming contract shared by the measure window and remote screenshots.

``DialogName`` is the wire enum a remote caller passes to ``dialog.screenshot``
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
from typing import TYPE_CHECKING

from zcu_tools.gui.expected_error import InvalidInputError

if TYPE_CHECKING:
    from zcu_tools.gui.session.ui.inspect_dialog import InspectPage


class DialogName(str, Enum):
    """Wire-stable identifiers for remotely controllable dialogs."""

    SETUP = "setup"
    DEVICE = "device"
    PREDICTOR = "predictor"
    INSPECT = "inspect"
    ARB_WAVEFORM = "arb_waveform"


def parse_dialog_name(value: object) -> DialogName:
    """Coerce a wire string into a ``DialogName`` enum.

    Accepts the lowercase wire form (``"setup"``) and the upper-case enum
    name (``"SETUP"``) for client ergonomics. Raises ``InvalidInputError`` if
    the name is unknown.
    """
    if isinstance(value, DialogName):
        return value
    if not isinstance(value, str):
        raise InvalidInputError(
            f"dialog name must be a string, got {type(value).__name__}"
        )
    lowered = value.lower()
    for name in DialogName:
        if name.value == lowered:
            return name
    upper = value.upper()
    for name in DialogName:
        if name.name == upper:
            return name
    raise InvalidInputError(f"unknown dialog name: {value!r}")


INSPECT_PAGES: tuple[InspectPage, ...] = ("parameters", "modules", "aliases")
"""Wire names of the Inspect tabs a dialog.screenshot may select."""


def parse_inspect_page(value: object) -> InspectPage:
    """Return the Inspect page named by the wire string value.

    Raises InvalidInputError for anything outside INSPECT_PAGES; no case
    folding or fallback page is applied.
    """
    for page in INSPECT_PAGES:
        if value == page:
            return page
    raise InvalidInputError(
        f"unknown Inspect tab: {value!r}; expected one of {list(INSPECT_PAGES)}"
    )


__all__ = ["INSPECT_PAGES", "DialogName", "parse_dialog_name", "parse_inspect_page"]
