from __future__ import annotations

from qick.asm_v2 import Macro
from typing_extensions import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..ir.base import IRCompileMixin


class MetaMacro(Macro):
    """A macro that emits a meta instruction for the IR builder."""

    # fields: type (str), name (str), info (dict)
    def __init__(
        self, type: str, name: str, info: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(type=type, name=name, info=info or {})

    def translate(self, prog: IRCompileMixin):  # type: ignore[override]
        prog._add_meta(type=self.type, name=self.name, info=self.info)
