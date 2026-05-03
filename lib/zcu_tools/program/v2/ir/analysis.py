from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from .instructions import (
    DmemReadInst,
    DmemWriteInst,
    Instruction,
    LabelInst,
    MetaInst,
    PortWriteInst,
)
from .node import InstNode, IRBranch, IRLoop, IRNode

if TYPE_CHECKING:
    from .pipeline import PipeLineConfig


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
    return dataclasses.replace(inst, annotations={})


def estimate_body_cost(body: list["IRNode"], config: PipeLineConfig) -> int:
    """Estimate the cycle cost of executing a sequence of IR nodes."""

    cost = 0
    for node in body:
        if isinstance(node, InstNode):
            inst = node.inst
            if isinstance(inst, PortWriteInst):
                cost += config.cost_wmem
            elif isinstance(inst, (DmemReadInst, DmemWriteInst)):
                cost += config.cost_dmem
            elif isinstance(inst, (MetaInst, LabelInst)):
                pass  # No runtime cycle cost
            else:
                cost += config.cost_default
        elif isinstance(node, IRLoop):
            # Loop iteration takes: testing counter (default), jump back (jump_flush), and body

            loop_overhead = 2 * config.cost_default + config.cost_jump_flush
            # If nested loop has constant n, multiply inner cost + loop overhead
            inner_cost = estimate_body_cost(node.body.insts, config)

            if isinstance(node.n, int):
                cost += node.n * (inner_cost + loop_overhead)
            elif node.range_hint is not None:
                cost += node.range_hint[1] * (inner_cost + loop_overhead)
            else:
                cost += inner_cost + loop_overhead
        elif isinstance(node, IRBranch):
            # Dispatch overhead: binary tree depth is ceil(log2(n)), each level costs
            # one TEST + one JUMP (2 * cost_default) plus a final unconditional JUMP.

            n_cases = len(node.cases)
            dispatch_depth = math.ceil(math.log2(n_cases)) if n_cases > 1 else 0
            dispatch_overhead = dispatch_depth * (
                2 * config.cost_default + config.cost_jump_flush
            )
            case_cost = max(
                (estimate_body_cost(case.insts, config) for case in node.cases),
                default=0,
            )
            cost += dispatch_overhead + case_cost

    return cost
