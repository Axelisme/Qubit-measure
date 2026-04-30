from __future__ import annotations

from qick.asm_v2 import Macro


class MetaMacro(Macro):
    """A macro that emits a meta instruction for the IR builder."""

    # fields: type (str), name (str)
    def translate(self, prog):  # type: ignore[override]
        # Emit a meta instruction without incrementing the instruction address.
        prog._add_asm(
            {
                "CMD": "__META__",
                "TYPE": self.type,
                "NAME": self.name,
                "ARGS": getattr(self, "args", {}),
            },
            0,
        )
