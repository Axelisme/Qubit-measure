from __future__ import annotations

from typing_extensions import TYPE_CHECKING

from .hw_semantics import TIMED_BASE_REG
from .instructions import (
    BaseInst,
    DmemReadInst,
    DmemWriteInst,
    LabelInst,
    MetaInst,
    PortWriteInst,
    RegWriteInst,
    TimeInst,
    WmemWriteInst,
)
from .node import BasicBlockNode, BlockNode, IRBranch, IRLoop, IRNode

if TYPE_CHECKING:
    from .pipeline import PipeLineConfig


from .operands import Literal


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
                    if isinstance(inst.lit, Literal) and inst.lit.value.startswith("#"):
                        try:
                            total += int(inst.lit.value[1:])
                        except ValueError:
                            continue
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
            case_ticks = [
                estimate_body_scheduled_ticks(case.insts) for case in node.cases
            ]
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
                n = 1  # unknown: underestimate to keep budget safe
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
            case_words = sum(estimate_flat_size(case.insts) for case in node.cases)
            size += setup_words + table_words + case_words
    return size


def instruction_reads(inst: BaseInst) -> set[str]:
    """Extract all registers read by an instruction."""
    return set(inst.reg_read)


def instruction_writes(inst: BaseInst) -> set[str]:
    """Extract all registers written by an instruction."""
    return set(inst.reg_write)


def reads_register(inst: BaseInst, reg: str) -> bool:
    """True if `reg` appears in inst.reg_read."""
    return reg in instruction_reads(inst)


def reads_implicit_time_base(inst: BaseInst) -> bool:
    """True if inst implicitly depends on s14 (timed writes, WAIT time, …).

    Use this in passes that move/insert TIME inc_ref to avoid crossing a
    timed-write barrier, instead of listing isinstance whitelists.
    """
    return TIMED_BASE_REG in instruction_reads(inst)


def is_wmem_load(inst: BaseInst) -> bool:
    """True for `REG_WR r_wave wmem [&addr]` — has dmem read side effect."""
    return isinstance(inst, RegWriteInst) and inst.src == "wmem"


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
                (estimate_body_cost(case.insts, config) for case in node.cases),
                default=0,
            )
            cost += dispatch_overhead + case_cost

    return cost
