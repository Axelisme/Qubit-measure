from __future__ import annotations

from typing import TYPE_CHECKING

from .instructions import Instruction

if TYPE_CHECKING:
    from .node import IRNode
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
    import dataclasses

    return dataclasses.replace(inst, annotations={})


def estimate_body_cost(body: list["IRNode"], config: "PipeLineConfig") -> int:
    """Estimate the cycle cost of executing a sequence of IR nodes."""
    from .node import InstNode, IRLoop, IRBranch
    from .instructions import (
        PortWriteInst,
        DmemReadInst,
        DmemWriteInst,
        MetaInst,
        LabelInst,
    )

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
            # If nested loop has constant n, multiply inner cost + loop overhead
            if isinstance(node.n, int):
                inner_cost = estimate_body_cost(node.body.insts, config)
                # Loop iteration takes: testing counter (default), jump back (jump_flush), and body
                loop_overhead = 2 * config.cost_default + config.cost_jump_flush
                cost += node.n * (inner_cost + loop_overhead)
            else:
                # Unknown bounds, assume heavily conservative cost
                cost += 100 * config.cost_default
        elif isinstance(node, IRBranch):
            # Base cost for dispatch plus average case cost
            dispatch_cost = config.cost_default * len(node.dispatch.insts)
            if node.cases:
                avg_case_cost = sum(estimate_body_cost(c.insts, config) for c in node.cases) // len(node.cases)
            else:
                avg_case_cost = 0
            cost += dispatch_cost + config.cost_jump_flush + avg_case_cost

    return cost
