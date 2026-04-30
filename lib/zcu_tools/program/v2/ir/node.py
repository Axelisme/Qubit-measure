from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IRNode:
    insts: list[dict]
    labels: dict[str, str]
