from __future__ import annotations

from dataclasses import dataclass

from qick.asm_v2 import AsmInst


@dataclass(frozen=True)
class IRNode:
    insts: list[AsmInst]
    labels: dict[str, str]
