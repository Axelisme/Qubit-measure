from __future__ import annotations

from dataclasses import dataclass

from .instructions import Instruction


@dataclass(frozen=True)
class IRNode:
    insts: list[Instruction]
    labels: dict[str, str]
