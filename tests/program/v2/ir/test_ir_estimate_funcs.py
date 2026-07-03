"""Tests for ir/analysis.py: estimate_body_scheduled_ticks, estimate_flat_size,
estimate_body_cost.

All three functions are used by UnrollLoopPass to drive the k-selection
heuristic.  Branches covered here:
- estimate_body_scheduled_ticks: lit inc_ref accumulation, register-driven
  skipped, IRLoop multiplier, IRLoop range_hint lower-bound, IRLoop unknown
  (no range_hint → 0), IRBranch min-across-cases pessimistic.
- estimate_flat_size: BasicBlockNode, IRLoop with int n / range_hint / unknown,
  IRBranch setup_words+table_words+case_words.
- estimate_body_cost: PortWriteInst→cost_wmem, DmemReadInst→cost_dmem,
  IRLoop with n/range_hint/unknown, IRBranch dispatch overhead + max case.
"""

from __future__ import annotations

from zcu_tools.program.v2.ir.analysis import (
    estimate_body_cost,
    estimate_body_scheduled_ticks,
    estimate_flat_size,
)
from zcu_tools.program.v2.ir.instructions import (
    DmemReadInst,
    JumpInst,
    NopInst,
    PortWriteInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label, LabelRef
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import (
    Immediate,
    ImmValue,
    MemAddr,
    Register,
    TimeOffset,
)
from zcu_tools.program.v2.ir.pipeline import PipeLineConfig


def _cfg(**kwargs) -> PipeLineConfig:
    return PipeLineConfig(**kwargs)


def _time_bb(lit: int) -> BasicBlockNode:
    return BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(lit))])


def _reg_time_bb(reg: str = "r1") -> BasicBlockNode:
    return BasicBlockNode(insts=[TimeInst(c_op="inc_ref", r1=Register(reg))])


def _loop(body: BasicBlockNode, n: int | Register, name: str = "lp") -> IRLoop:
    return IRLoop(
        name=name,
        counter_reg=Register("r0"),
        n=n,
        body=BlockNode(insts=[body]),
    )


def _branch(*case_bbs: BasicBlockNode, name: str = "br") -> IRBranch:
    return IRBranch(
        name=name,
        compare_reg=Register("r_sel"),
        cases=[BlockNode(insts=[bb]) for bb in case_bbs],
    )


# ---------------------------------------------------------------------------
# estimate_body_scheduled_ticks
# ---------------------------------------------------------------------------


def test_scheduled_ticks_empty_body():
    assert estimate_body_scheduled_ticks([]) == 0


def test_scheduled_ticks_lit_inc_ref_accumulates():
    bbs: list[IRNode] = [_time_bb(100), _time_bb(50)]
    assert estimate_body_scheduled_ticks(bbs) == 150


def test_scheduled_ticks_register_driven_contributes_zero():
    bbs: list[IRNode] = [_time_bb(100), _reg_time_bb("r1")]
    assert estimate_body_scheduled_ticks(bbs) == 100


def test_scheduled_ticks_irloop_const_n_multiplies():
    inner = _time_bb(200)
    loop = _loop(inner, n=3)
    assert estimate_body_scheduled_ticks([loop]) == 600


def test_scheduled_ticks_irloop_range_hint_uses_lower_bound():
    inner = _time_bb(100)
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        range_hint=(2, 8),
        body=BlockNode(insts=[inner]),
    )
    assert estimate_body_scheduled_ticks([loop]) == 200


def test_scheduled_ticks_irloop_no_range_hint_contributes_zero():
    inner = _time_bb(100)
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        body=BlockNode(insts=[inner]),
    )
    assert estimate_body_scheduled_ticks([loop]) == 0


def test_scheduled_ticks_irbranch_pessimistic_min():
    fast = BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(50))])
    slow = BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(200))])
    branch = _branch(fast, slow)
    assert estimate_body_scheduled_ticks([branch]) == 50


def test_scheduled_ticks_irbranch_accepts_basic_block_cases():
    fast = BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(50))])
    slow = BasicBlockNode(insts=[TimeInst(c_op="inc_ref", lit=Immediate(200))])
    branch = IRBranch(
        name="br",
        compare_reg=Register("r_sel"),
        cases=[fast, slow],
    )
    assert estimate_body_scheduled_ticks([branch]) == 50


def test_scheduled_ticks_irbranch_accepts_nested_loop_case():
    case = IRLoop(
        name="case_loop",
        counter_reg=Register("r0"),
        n=2,
        body=BlockNode(insts=[_time_bb(30)]),
    )
    branch = IRBranch(name="br", compare_reg=Register("r_sel"), cases=[case])
    assert estimate_body_scheduled_ticks([branch]) == 60


def test_scheduled_ticks_irbranch_no_cases_contributes_zero():
    branch = IRBranch(name="br", compare_reg=Register("r0"), cases=[])
    assert estimate_body_scheduled_ticks([branch]) == 0


def test_scheduled_ticks_nested_block_node():
    inner = _time_bb(75)
    wrapper = BlockNode(insts=[inner])
    assert estimate_body_scheduled_ticks([wrapper]) == 75


# ---------------------------------------------------------------------------
# estimate_flat_size
# ---------------------------------------------------------------------------


def test_flat_size_empty_body():
    assert estimate_flat_size([]) == 0


def test_flat_size_basic_block_with_branch():
    bb = BasicBlockNode(
        insts=[NopInst()],
        branch=JumpInst(label=LabelRef(Label("x"))),
    )
    size = estimate_flat_size([bb])
    assert size == 2  # NopInst(addr_inc=1) + JumpInst(addr_inc=1)


def test_flat_size_irloop_const_n():
    inner = BasicBlockNode(insts=[NopInst()])
    loop = _loop(inner, n=4)
    size = estimate_flat_size([loop])
    # formula: 2 + n * inner_size = 2 + 4*1 = 6
    assert size == 6


def test_flat_size_irloop_range_hint_uses_upper_bound():
    inner = BasicBlockNode(insts=[NopInst()])
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        range_hint=(1, 8),
        body=BlockNode(insts=[inner]),
    )
    size = estimate_flat_size([loop])
    # 2 + 8*1 = 10
    assert size == 10


def test_flat_size_irloop_unknown_uses_n1():
    inner = BasicBlockNode(insts=[NopInst()])
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        body=BlockNode(insts=[inner]),
    )
    size = estimate_flat_size([loop])
    # Unknown dynamic loops stay rolled: 2 loop overhead words + one body copy.
    assert size == 3


def test_flat_size_irbranch_accounts_all_terms():
    case0 = BasicBlockNode(insts=[NopInst()])
    case1 = BasicBlockNode(insts=[NopInst(), NopInst()])
    branch = _branch(case0, case1)
    size = estimate_flat_size([branch])
    # setup=4, table=2*2=4, cases=1+2=3 → 11
    assert size == 11


def test_flat_size_irbranch_accepts_basic_block_cases():
    case0 = BasicBlockNode(insts=[NopInst()])
    case1 = BasicBlockNode(insts=[NopInst(), NopInst()])
    branch = IRBranch(
        name="br",
        compare_reg=Register("r_sel"),
        cases=[case0, case1],
    )
    assert estimate_flat_size([branch]) == 11


def test_flat_size_nested_block_node():
    inner = BasicBlockNode(insts=[NopInst()])
    wrapper = BlockNode(insts=[inner])
    assert estimate_flat_size([wrapper]) == 1


# ---------------------------------------------------------------------------
# estimate_body_cost
# ---------------------------------------------------------------------------


def test_body_cost_empty():
    cfg = _cfg()
    assert estimate_body_cost([], cfg) == 0


def test_body_cost_port_write_uses_cost_wmem():
    bb = BasicBlockNode(
        insts=[
            PortWriteInst(
                dst=ImmValue(2),
                time=TimeOffset(0),
                addr=MemAddr(0),
            )
        ]
    )
    cfg = _cfg(cost_wmem=10, cost_default=1)
    assert estimate_body_cost([bb], cfg) == 10


def test_body_cost_dmem_read_uses_cost_dmem():
    bb = BasicBlockNode(insts=[DmemReadInst(dst=Register("r0"), addr=Register("r1"))])
    cfg = _cfg(cost_dmem=5, cost_default=1)
    assert estimate_body_cost([bb], cfg) == 5


def test_body_cost_nop_uses_cost_default():
    bb = BasicBlockNode(insts=[NopInst()])
    cfg = _cfg(cost_default=2)
    assert estimate_body_cost([bb], cfg) == 2


def test_body_cost_branch_counts():
    bb = BasicBlockNode(
        insts=[NopInst()],
        branch=JumpInst(label=LabelRef(Label("x"))),
    )
    cfg = _cfg(cost_default=1)
    assert estimate_body_cost([bb], cfg) == 2  # inst + branch


def test_body_cost_irloop_const_n_multiplies():
    inner = BasicBlockNode(insts=[NopInst()])
    loop = _loop(inner, n=3)
    cfg = _cfg(cost_default=1, cost_jump_flush=2)
    inner_cost = 1
    loop_overhead = 1 + 2
    assert estimate_body_cost([loop], cfg) == 3 * (inner_cost + loop_overhead)


def test_body_cost_irloop_range_hint_uses_upper_bound():
    inner = BasicBlockNode(insts=[NopInst()])
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        range_hint=(1, 5),
        body=BlockNode(insts=[inner]),
    )
    cfg = _cfg(cost_default=1, cost_jump_flush=2)
    inner_cost = 1
    loop_overhead = 1 + 2
    assert estimate_body_cost([loop], cfg) == 5 * (inner_cost + loop_overhead)


def test_body_cost_irloop_unknown_uses_single_iteration():
    inner = BasicBlockNode(insts=[NopInst()])
    loop = IRLoop(
        name="lp",
        counter_reg=Register("r0"),
        n=Register("n_reg"),
        body=BlockNode(insts=[inner]),
    )
    cfg = _cfg(cost_default=1, cost_jump_flush=2)
    inner_cost = 1
    loop_overhead = 1 + 2
    assert estimate_body_cost([loop], cfg) == inner_cost + loop_overhead


def test_body_cost_irbranch_dispatch_overhead_plus_max_case():
    fast = BasicBlockNode(insts=[NopInst()])
    slow = BasicBlockNode(insts=[NopInst(), NopInst()])
    branch = _branch(fast, slow)
    cfg = _cfg(cost_default=1, cost_jump_flush=2)
    dispatch_overhead = 4 * 1 + 2 * (1 + 2)
    max_case = 2
    assert estimate_body_cost([branch], cfg) == dispatch_overhead + max_case


def test_body_cost_irbranch_accepts_basic_block_cases():
    fast = BasicBlockNode(insts=[NopInst()])
    slow = BasicBlockNode(insts=[NopInst(), NopInst()])
    branch = IRBranch(
        name="br",
        compare_reg=Register("r_sel"),
        cases=[fast, slow],
    )
    cfg = _cfg(cost_default=1, cost_jump_flush=2)
    dispatch_overhead = 4 * 1 + 2 * (1 + 2)
    assert estimate_body_cost([branch], cfg) == dispatch_overhead + 2


def test_body_cost_irbranch_no_cases_only_dispatch_overhead():
    branch = IRBranch(name="br", compare_reg=Register("r0"), cases=[])
    cfg = _cfg(cost_default=1, cost_jump_flush=2)
    dispatch_overhead = 4 * 1 + 2 * (1 + 2)
    assert estimate_body_cost([branch], cfg) == dispatch_overhead
