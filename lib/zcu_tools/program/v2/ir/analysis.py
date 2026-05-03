from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from .instructions import (
    DmemReadInst,
    DmemWriteInst,
    Instruction,
    LabelInst,
    MetaInst,
    PortWriteInst,
    TimeInst,
)
from .node import InstNode, IRBranch, IRLoop, IRNode

if TYPE_CHECKING:
    from .pipeline import PipeLineConfig


def estimate_body_scheduled_ticks(body: list["IRNode"]) -> Optional[int]:
    """Sum of literal inc_ref delay ticks in a body sequence.

    Returns the total tProc clock cycles reserved for scheduled IO per loop
    iteration, or None if any dynamic (register-driven) delay is encountered
    at any depth — caller should fallback to no-unroll in that case.
    """
    total = 0
    for node in body:
        if isinstance(node, InstNode):
            inst = node.inst
            if isinstance(inst, TimeInst) and inst.c_op == "inc_ref":
                if inst.r1 is not None:
                    return None  # dynamic delay — cannot analyze statically
                if inst.lit is not None and inst.lit.startswith("#"):
                    try:
                        total += int(inst.lit[1:])
                    except ValueError:
                        return None
        elif isinstance(node, IRLoop):
            inner = estimate_body_scheduled_ticks(node.body.insts)
            if inner is None:
                return None
            if isinstance(node.n, int):
                multiplier = node.n
            elif node.range_hint is not None:
                multiplier = node.range_hint[1]  # upper bound (conservative)
            else:
                return None  # unknown iteration count
            total += multiplier * inner
        elif isinstance(node, IRBranch):
            # Pessimistic: shortest IO window across all cases
            case_ticks = [estimate_body_scheduled_ticks(case.insts) for case in node.cases]
            if any(t is None for t in case_ticks):
                return None
            total += min(t for t in case_ticks)  # type: ignore[type-var]
    return total if total > 0 else None


def estimate_flat_size(nodes: list["IRNode"]) -> int:
    """Estimate the number of pmem words emitted by a node sequence.

    Used to check how many unrolled copies fit within the pmem budget before
    deciding the unroll factor k.
    """
    size = 0
    for node in nodes:
        if isinstance(node, InstNode):
            inst = node.inst
            if isinstance(inst, LabelInst):
                pass  # labels occupy no pmem slot
            else:
                size += inst.addr_inc
        elif isinstance(node, IRLoop):
            inner = estimate_flat_size(node.body.insts)
            if isinstance(node.n, int):
                n = node.n
            elif node.range_hint is not None:
                n = node.range_hint[1]
            else:
                n = 1  # unknown: underestimate to keep budget safe
            size += 2 + n * inner  # TEST + JUMP_back + n * body
        elif isinstance(node, IRBranch):
            n_cases = len(node.cases)
            dispatch_depth = math.ceil(math.log2(n_cases)) if n_cases > 1 else 0
            dispatch_words = dispatch_depth * 2  # TEST + JUMP per level
            case_size = max((estimate_flat_size(case.insts) for case in node.cases), default=0)
            size += dispatch_words + case_size
    return size


def instruction_reads(inst: Instruction) -> set[str]:
    """Extract all registers read by an instruction."""
    return set(inst.reg_read)


def instruction_writes(inst: Instruction) -> set[str]:
    """Extract all registers written by an instruction."""
    return set(inst.reg_write)


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
