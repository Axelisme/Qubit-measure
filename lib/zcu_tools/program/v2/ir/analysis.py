from __future__ import annotations

import re
from typing import Any

from .instructions import GenericInst, Instruction

_REG_TOKEN_RE = re.compile(r"\b[srw]\d+\b|s_[A-Za-z0-9_]+|temp_reg_\d+")


def instruction_reads(inst: Instruction) -> set[str]:
    if not isinstance(inst, GenericInst):
        return set()

    reads: set[str] = set()
    for key in ("R1", "R2", "R3", "ADDR"):
        value = inst.args.get(key)
        if isinstance(value, str) and not value.startswith("#"):
            reads.add(value)

    for key in ("SRC", "OP", "TIME"):
        reads.update(_regs_from_value(inst.args.get(key)))

    return reads


def instruction_writes(inst: Instruction) -> set[str]:
    if not isinstance(inst, GenericInst):
        return set()

    writes: set[str] = set()
    for key in ("DST", "WR"):
        value = inst.args.get(key)
        if isinstance(value, str):
            writes.add(_strip_write_modifier(value))
    return writes


def is_marked_hoistable(inst: Instruction) -> bool:
    return isinstance(inst, GenericInst) and inst.args.get("IR_HOISTABLE") is True


def strip_internal_annotations(inst: Instruction) -> Instruction:
    if not isinstance(inst, GenericInst):
        return inst
    cleaned = {
        key: value
        for key, value in inst.args.items()
        if not key.startswith("IR_")
    }
    if cleaned == inst.args:
        return inst
    return GenericInst(
        cmd=inst.cmd,
        args=cleaned,
        line=inst.line,
        p_addr=inst.p_addr,
    )


def _regs_from_value(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    if value.startswith("#"):
        return set()
    return set(_REG_TOKEN_RE.findall(value))


def _strip_write_modifier(value: str) -> str:
    return value.split(" ", 1)[0]
