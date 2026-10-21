from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qick.asm_v2 import Macro

if TYPE_CHECKING:
    from ..ir.base import IRCompileMixin


class MetaMacro(Macro):
    """A macro that emits a meta instruction for the IR builder.

    regs maps info keys to register names resolved by the compiler at translate
    time. Use it for logical names (e.g. a loop name like "reset_sel"), ASM
    addresses or QICK aliases. Resolved addresses overwrite matching info keys
    without mutating the inputs.
    """

    # fields: type (str), name (str), info (dict), regs (dict)
    def __init__(
        self,
        type: str,
        name: str,
        info: dict[str, Any] | None = None,
        regs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(type=type, name=name, info=info or {}, regs=regs or {})

    def translate(self, prog: IRCompileMixin):
        prog.record_meta(type=self.type, name=self.name, info=self.info, regs=self.regs)
