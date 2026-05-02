from __future__ import annotations

from qick.asm_v2 import Macro
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir.base import IRCompileMixin


class MetaMacro(Macro):
    """A macro that emits a meta instruction for the IR builder."""

    # fields: type (str), name (str)
    def translate(self, prog: IRCompileMixin):  # type: ignore[override]
        # Emit a meta structure marker directly into the program's tracker.
        prog._add_meta(
            type=self.type,
            name=self.name,
            info=getattr(self, "args", {}),
        )
