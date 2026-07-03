from __future__ import annotations

from typing import TYPE_CHECKING

from .instructions import (
    BaseInst,
    DmemReadInst,
    DmemWriteInst,
    LabelInst,
    MetaInst,
    PortWriteInst,
    TimeInst,
    WmemWriteInst,
)
from .labels import Label
from .node import BasicBlockNode, BlockNode, IRBranch, IRLoop, IRNode
from .operands import Immediate

if TYPE_CHECKING:
    from .pipeline import ChunkList, PipeLineConfig


def collect_referenced_labels(chunks: ChunkList) -> set[Label]:
    """Collect all Labels referenced across a chunk list.

    Uses ``BaseInst.need_labels`` so multi-label references (e.g. a dmem
    dispatch table addressed via ``DmemAddr``) keep every referenced label
    alive, not just the single ``need_label``.
    """
    refs: set[Label] = set()
    for chunk in chunks:
        if not isinstance(chunk, BasicBlockNode):
            continue
        for inst in (
            *chunk.labels,
            *chunk.insts,
            *([chunk.branch] if chunk.branch else []),
        ):
            if isinstance(inst, BaseInst):
                refs |= inst.need_labels
    return refs


def estimate_body_scheduled_ticks(body: list[IRNode]) -> int:
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
                    if isinstance(inst.lit, Immediate):
                        total += inst.lit.value
        elif isinstance(node, BlockNode):
            total += estimate_body_scheduled_ticks(node.insts)
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
            case_ticks = [estimate_body_scheduled_ticks([case]) for case in node.cases]
            if case_ticks:
                total += min(case_ticks)
    return total


def estimate_flat_size(nodes: list[IRNode]) -> int:
    """Estimate the number of pmem words emitted by a node sequence.

    Used to check how many unrolled copies fit within the pmem budget before
    deciding the unroll factor k.
    """
    size = 0
    for node in nodes:
        if isinstance(node, BasicBlockNode):
            for inst in node.insts:
                if isinstance(inst, BaseInst):
                    size += inst.addr_inc
            if node.branch is not None:
                size += node.branch.addr_inc
        elif isinstance(node, BlockNode):
            size += estimate_flat_size(node.insts)
        elif isinstance(node, IRLoop):
            inner = estimate_flat_size(node.body.insts)
            if isinstance(node.n, int):
                n = node.n
            elif node.range_hint is not None:
                n = node.range_hint[1]
            else:
                # Unknown dynamic loops stay rolled, so one physical body copy is the
                # best flat-size approximation.
                n = 1
            # IRLoop.body is treated as one full logical iteration, including
            # the counter update even if later optimizers move or merge it.
            # Shape: [guard? 1] + init 1 + n * inner + cond-back 1.
            size += 2 + n * inner
        elif isinstance(node, IRBranch):
            # Conservative PMEM estimate for dispatch-table lowering:
            # - setup: REG_WR s15 label table_0 + up to two adds + JUMP s15
            # - table: worst-case 2 words per entry (big-PMEM mode)
            # - bodies: all cases are emitted physically, so sum their sizes
            setup_words = 4
            table_words = 2 * len(node.cases)
            case_words = sum(estimate_flat_size([case]) for case in node.cases)
            size += setup_words + table_words + case_words
    return size


def estimate_body_cost(body: list[IRNode], config: PipeLineConfig) -> int:
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
        elif isinstance(node, IRLoop):
            # Counter update cost stays inside inner_cost because it exists
            # regardless of whether the loop is unrolled. loop_overhead models
            # only the single condensed back-edge JUMP plus its flush penalty.
            loop_overhead = config.cost_default + config.cost_jump_flush
            inner_cost = estimate_body_cost(node.body.insts, config)

            if isinstance(node.n, int):
                cost += node.n * (inner_cost + loop_overhead)
            elif node.range_hint is not None:
                cost += node.range_hint[1] * (inner_cost + loop_overhead)
            else:
                cost += inner_cost + loop_overhead
        elif isinstance(node, IRBranch):
            # Constant-depth dispatch-table runtime cost:
            # setup (REG_WR/ALU ops) + indirect jump + stub jump.
            dispatch_overhead = 4 * config.cost_default + 2 * (
                config.cost_default + config.cost_jump_flush
            )
            case_cost = max(
                (estimate_body_cost([case], config) for case in node.cases),
                default=0,
            )
            cost += dispatch_overhead + case_cost

    return cost
