from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .instructions import (
    DmemReadInst,
    DmemWriteInst,
    Instruction,
    LabelInst,
    MetaInst,
    PortWriteInst,
    TimeInst,
    WmemWriteInst,
)
from .node import BasicBlockNode, BlockNode, InstNode, IRBranch, IRLoop, IRNode

if TYPE_CHECKING:
    from .pipeline import PipeLineConfig


def estimate_body_scheduled_ticks(body: list["IRNode"]) -> int:
    """Lower-bound on inc_ref delay ticks in a body sequence.

    Dynamic (register-driven) `inc_ref` contributes 0 to the total — the
    estimate represents the *guaranteed* scheduled IO window, not the
    expected one. This lets loops with mixed literal + dynamic delays still
    be analyzed for unroll based on their literal budget.
    """
    total = 0
    for node in body:
        if isinstance(node, BasicBlockNode):
            for inst in node.insts:
                if isinstance(inst, TimeInst) and inst.c_op == "inc_ref":
                    if inst.r1 is not None:
                        continue
                    if inst.lit is not None and inst.lit.startswith("#"):
                        try:
                            total += int(inst.lit[1:])
                        except ValueError:
                            continue
        elif isinstance(node, BlockNode):
            total += estimate_body_scheduled_ticks(node.insts)
        elif isinstance(node, InstNode):
            inst = node.inst
            if isinstance(inst, TimeInst) and inst.c_op == "inc_ref":
                if inst.r1 is not None:
                    continue  # dynamic delay contributes 0 (lower bound)
                if inst.lit is not None and inst.lit.startswith("#"):
                    try:
                        total += int(inst.lit[1:])
                    except ValueError:
                        continue
        elif isinstance(node, IRLoop):
            inner = estimate_body_scheduled_ticks(node.body.insts)
            if isinstance(node.n, int):
                multiplier = node.n
            elif node.range_hint is not None:
                multiplier = node.range_hint[0]  # lower bound (guaranteed)
            else:
                continue  # unknown iteration count — contributes 0
            total += multiplier * inner
        elif isinstance(node, IRBranch):
            # Pessimistic: shortest IO window across all cases
            case_ticks = [
                estimate_body_scheduled_ticks(case.insts) for case in node.cases
            ]
            if case_ticks:
                total += min(case_ticks)
    return total


def estimate_flat_size(nodes: list["IRNode"]) -> int:
    """Estimate the number of pmem words emitted by a node sequence.

    Used to check how many unrolled copies fit within the pmem budget before
    deciding the unroll factor k.
    """
    size = 0
    for node in nodes:
        if isinstance(node, BasicBlockNode):
            for inst in node.insts:
                if not isinstance(inst, (LabelInst, MetaInst)):
                    size += inst.addr_inc
            if node.branch is not None:
                size += node.branch.addr_inc
        elif isinstance(node, BlockNode):
            size += estimate_flat_size(node.insts)
        elif isinstance(node, InstNode):
            inst = node.inst
            if not isinstance(inst, LabelInst):
                size += inst.addr_inc
        elif isinstance(node, IRLoop):
            inner = estimate_flat_size(node.body.insts)
            if isinstance(node.n, int):
                n = node.n
            elif node.range_hint is not None:
                n = node.range_hint[1]
            else:
                n = 1  # unknown: underestimate to keep budget safe
            # Shape: [guard? 2] + init 1 + n * (inner + i++ 1) + cond-back 1.
            # Guard only emitted for runtime-driven n; ignored here for the
            # constant case that drives most budgets.
            size += 2 + n * (inner + 1)
        elif isinstance(node, IRBranch):
            n_cases = len(node.cases)
            dispatch_depth = math.ceil(math.log2(n_cases)) if n_cases > 1 else 0
            dispatch_words = dispatch_depth * 2  # TEST + JUMP per level
            case_size = max(
                (estimate_flat_size(case.insts) for case in node.cases), default=0
            )
            size += dispatch_words + case_size
    return size


def instruction_reads(inst: Instruction) -> set[str]:
    """Extract all registers read by an instruction."""
    return set(inst.reg_read)


def instruction_writes(inst: Instruction) -> set[str]:
    """Extract all registers written by an instruction."""
    return set(inst.reg_write)


def estimate_body_cost(body: list["IRNode"], config: "PipeLineConfig") -> int:
    """Estimate the cycle cost of executing a sequence of IR nodes."""

    cost = 0
    for node in body:
        if isinstance(node, BasicBlockNode):
            for inst in node.insts:
                if isinstance(inst, (PortWriteInst, WmemWriteInst)):
                    cost += config.cost_wmem
                elif isinstance(inst, (DmemReadInst, DmemWriteInst)):
                    cost += config.cost_dmem
                elif isinstance(inst, (MetaInst, LabelInst)):
                    pass
                else:
                    cost += config.cost_default
            if node.branch is not None:
                cost += config.cost_default
        elif isinstance(node, BlockNode):
            cost += estimate_body_cost(node.insts, config)
        elif isinstance(node, InstNode):
            inst = node.inst
            if isinstance(inst, (PortWriteInst, WmemWriteInst)):
                cost += config.cost_wmem
            elif isinstance(inst, (DmemReadInst, DmemWriteInst)):
                cost += config.cost_dmem
            elif isinstance(inst, (MetaInst, LabelInst)):
                pass
            else:
                cost += config.cost_default
        elif isinstance(node, IRLoop):
            loop_overhead = 2 * config.cost_default + config.cost_jump_flush
            inner_cost = estimate_body_cost(node.body.insts, config)

            if isinstance(node.n, int):
                cost += node.n * (inner_cost + loop_overhead)
            elif node.range_hint is not None:
                cost += node.range_hint[1] * (inner_cost + loop_overhead)
            else:
                cost += inner_cost + loop_overhead
        elif isinstance(node, IRBranch):
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
