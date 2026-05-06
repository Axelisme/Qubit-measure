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
from .node import BlockNode, InstNode, IRBranch, IRJumpTableLoop, IRLoop, IRNode

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
        if isinstance(node, BlockNode):
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
        elif isinstance(node, IRJumpTableLoop):
            # n_reg is dynamic; lower bound on guaranteed scheduled IO is 0.
            pass
    return total


def estimate_flat_size(nodes: list["IRNode"]) -> int:
    """Estimate the number of pmem words emitted by a node sequence.

    Used to check how many unrolled copies fit within the pmem budget before
    deciding the unroll factor k.
    """
    size = 0
    for node in nodes:
        if isinstance(node, BlockNode):
            size += estimate_flat_size(node.insts)
        elif isinstance(node, InstNode):
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
        elif isinstance(node, IRJumpTableLoop):
            # See passes.loop_dispatch.emit_jump_table_loop for the exact
            # shape. Approximate count (labels and meta are 0 words):
            #   prologue (2) + k * (body_words + 1 i++) + back-edge (3)
            #   + dispatch (3 + shift_add(<= max_words)) + JUMP s15 (1)
            #   + fast_path (2)
            # We use a generous cap of 16 dispatch words as a reasonable
            # upper bound; budgets here are advisory.
            per_body = sum(estimate_flat_size(b.insts) for b in node.bodies)
            size += 2 + per_body + node.k + 3 + 16 + 1 + 2
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
        if isinstance(node, BlockNode):
            cost += estimate_body_cost(node.insts, config)
        elif isinstance(node, InstNode):
            inst = node.inst
            if isinstance(inst, (PortWriteInst, WmemWriteInst)):
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
        elif isinstance(node, IRJumpTableLoop):
            # Conservative lower bound on JT loop cost: one body execution
            # plus the jump-table back-edge / dispatch (a constant ~10
            # cycles). The runtime n is unknown so we cannot project a
            # better estimate.
            body_cost = max(
                (estimate_body_cost(b.insts, config) for b in node.bodies),
                default=0,
            )
            cost += body_cost + 10

    return cost
