from __future__ import annotations

import dataclasses
from typing import Any

from .instructions import (
    GenericInst,
    Instruction,
)
from .utils import regs_from_value, strip_write_modifier


def instruction_reads(inst: Instruction) -> set[str]:
    """Extract all registers read by an instruction."""
    return set(inst.reg_read)


def instruction_writes(inst: Instruction) -> set[str]:
    """Extract all registers written by an instruction."""
    return set(inst.reg_write)


def is_marked_hoistable(inst: Instruction) -> bool:
    return inst.annotations.get("IR_HOISTABLE") is True


def strip_internal_annotations(inst: Instruction) -> Instruction:
    if not inst.annotations:
        return inst
    
    # annotations already contains only IR_ fields (extracted in from_dict)
    # So to "strip" them, we just need to return a new instance with empty annotations.
    # However, since Instruction is frozen, we must use replace() or similar.
    # Actually, using dataclasses.replace is best.
    import dataclasses
    return dataclasses.replace(inst, annotations={})
