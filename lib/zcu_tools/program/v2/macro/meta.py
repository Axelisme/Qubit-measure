from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qick.asm_v2 import Macro

if TYPE_CHECKING:
    from ..ir.base import IRCompileMixin


class MetaMacro(Macro):
    """A macro that emits a meta instruction for the IR builder.

    regs maps info keys to register names that need to be resolved to hardware
    addresses at translate time via prog._get_reg().  This is necessary when
    the caller only knows a logical name (e.g. a loop name like "reset_sel")
    that QICK registers under reg_dict, not a bare ASM address like "r0".
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

    def translate(self, prog: IRCompileMixin):  # type: ignore[override]
        resolved_info = dict(self.info)
        for key, reg_name in self.regs.items():
            resolved_info[key] = prog._get_reg(reg_name)
        prog._add_meta(type=self.type, name=self.name, info=resolved_info)
