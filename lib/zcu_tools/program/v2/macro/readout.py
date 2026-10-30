"""Readout provenance in the emission stream, without hardware instructions."""

from collections.abc import Sequence

from qick.asm_v2 import Macro

from .loop import CloseInnerLoop, OpenInnerLoop
from .meta import MetaMacro


def readout_control_scope(macros: Sequence[Macro]) -> str | None:
    """Return the innermost Repeat/Branch enclosing a macro-stream prefix.

    macros ends immediately before the ADC trigger under consideration.
    Acquisition repetition/sweep loops are not Repeat and remain supported.
    """
    scopes: list[str] = []
    for macro in macros:
        if isinstance(macro, OpenInnerLoop):
            scopes.append(f"Repeat {macro.name!r}")
        elif isinstance(macro, CloseInnerLoop):
            scopes.pop()
        elif isinstance(macro, MetaMacro):
            if macro.type == "BRANCH_START":
                scopes.append(f"Branch {macro.name!r}")
            elif macro.type == "BRANCH_END":
                scopes.pop()
    return scopes[-1] if scopes else None


class ReadoutEmission(Macro):
    """Associate the following ADC trigger with its emitting module instance.

    owner is an opaque module identity, borrowed for the lifetime of the program.
    ro_ch is its logical ADC channel number (first tone for mux). label is the
    optional emit tag; None selects the owner's module name. This marker emits no instructions
    and does not change trigger counts, timing, or acquisition configuration.
    """

    def __init__(self, owner: object, ro_ch: int, label: str | None = None) -> None:
        """Capture the borrowed owner, first ADC and optional emission label."""
        super().__init__()
        self.owner = owner
        self.ro_ch = ro_ch
        self.label = label

    def translate(self, prog: object) -> None:
        """Leave the compiled program unchanged; provenance stays in macro_list."""
