from __future__ import annotations

from typing import cast

from zcu_tools.program.v2.ir.instructions import (
    JumpInst,
    LabelInst,
    NopInst,
    PortWriteInst,
    RegWriteInst,
    TestInst,
    TimeInst,
)
from zcu_tools.program.v2.ir.labels import Label
from zcu_tools.program.v2.ir.node import BlockNode, InstNode, IRLoop, IRNode, RootNode
from zcu_tools.program.v2.ir.passes import (
    DeadLabelEliminationPass,
    DeadWriteEliminationPass,
    UnrollSmallLoopPass,
)
from zcu_tools.program.v2.ir.pipeline import (
    PipeLineConfig,
    PipeLineContext,
    make_default_pipeline,
)


def _flat_inst_count(root: RootNode) -> int:
    """Count InstNodes after flattening BlockNodes recursively."""
    count = 0
    stack: list[IRNode] = list(root.insts)
    while stack:
        node = stack.pop()
        if isinstance(node, InstNode):
            count += 1
        elif isinstance(node, BlockNode):
            stack.extend(node.insts)
    return count


# ---------------------------------------------------------------------------
# Dead write / dead label tests (unchanged from prior phases)
# ---------------------------------------------------------------------------


def test_dead_write_elimination_removes_overwritten_write():
    root = RootNode(
        insts=[
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"})),
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"})),
            InstNode(NopInst()),
        ]
    )

    out = DeadWriteEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, RegWriteInst)
    assert getattr(cast(InstNode, out.insts[0]).inst, "extra_args")["LIT"] == "#2"
    assert isinstance(out.insts[1], InstNode)
    assert isinstance(cast(InstNode, out.insts[1]).inst, NopInst)


def test_dead_write_elimination_keeps_write_before_read():
    root = RootNode(
        insts=[
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"})),
            InstNode(TestInst(op="s1 - #1")),
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"})),
        ]
    )

    out = DeadWriteEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 3
    assert [
        getattr(cast(InstNode, item).inst, "extra_args").get("LIT")
        for item in out.insts
        if isinstance(item, InstNode) and isinstance(item.inst, RegWriteInst)
    ] == [
        "#1",
        "#2",
    ]


def test_dead_label_elimination_removes_unreferenced_label():
    root = RootNode(
        insts=[
            InstNode(LabelInst(name=Label("dead"))),
            InstNode(NopInst()),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], InstNode)
    assert isinstance(cast(InstNode, out.insts[0]).inst, NopInst)


def test_dead_label_elimination_keeps_referenced_label():
    l = Label("keep")
    root = RootNode(
        insts=[
            InstNode(LabelInst(name=l)),
            InstNode(JumpInst(label=l)),
        ]
    )

    out = DeadLabelEliminationPass().process(
        root, PipeLineContext(config=PipeLineConfig())
    )

    assert len(out.insts) == 2
    assert isinstance(cast(InstNode, out.insts[0]).inst, LabelInst)
    assert isinstance(cast(InstNode, out.insts[1]).inst, JumpInst)


def test_dead_label_elimination_keeps_pseudo_labels():
    """Pseudo labels (HERE, NEXT, PREV, SKIP) are assembler tokens, not IR labels."""
    for pseudo in ("HERE", "NEXT", "PREV", "SKIP"):
        root = RootNode(
            insts=[
                InstNode(LabelInst(name=Label(pseudo))),
                InstNode(NopInst()),
            ]
        )

        out = DeadLabelEliminationPass().process(
            root, PipeLineContext(config=PipeLineConfig())
        )

        assert len(out.insts) == 2, f"pseudo label {pseudo!r} was incorrectly removed"
        assert isinstance(cast(InstNode, out.insts[0]).inst, LabelInst)


# ---------------------------------------------------------------------------
# Unroll tests (Phase 8 — joint slack/budget formula)
# ---------------------------------------------------------------------------
#
# k formula recap:
#   slack    = scheduled_ticks - body_cost
#   k_timing = max_unroll_factor               if slack <= 0
#            = min(ceil(overhead/slack), cap)  otherwise
#   k_budget = pmem_budget // body_size        (if body_size > 0 and budget set)
#   k        = min(k_timing, k_budget)
#
# Default config: cost_default=1, cost_wmem=4, cost_dmem=4, cost_jump_flush=4,
#                 max_unroll_factor=8, pmem_budget=None.
# loop_overhead = 2*cost_default + cost_jump_flush = 6.


def test_unroll_full_expansion_when_n_le_k():
    """n=3 body=inc_ref #1: scheduled=1, body_cost=1, slack=0 → k_timing=8;
    no budget → k=8. n(=3) <= k → fully expand into BlockNode of 3 copies."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Expansion is wrapped in a single BlockNode; flatten count should be 3.
    assert _flat_inst_count(out) == 3
    assert all(
        isinstance(item, InstNode) and isinstance(item.inst, TimeInst)
        for blk in out.insts
        for item in (blk.insts if isinstance(blk, BlockNode) else [blk])
    )


def test_unroll_full_expansion_preserves_internal_label():
    """Internal labels are deepcopied per body copy with unique names."""
    Label.reset()
    inner = Label.make_new("inner")
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[
                        InstNode(LabelInst(name=inner)),
                        InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                    ]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert _flat_inst_count(out) == 6


def test_unroll_partial_unroll_produces_loop_plus_remainder():
    """body=inc_ref #1: scheduled=1, body_cost=1, slack=0 → k_timing=cap=8.
    pmem_budget=8, body_size=1 → k_budget=8 → k=8.
    n=20: iters=2, remainder=4 → BlockNode containing IRLoop(n=2) + 4 copies."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # Top level should contain a single BlockNode
    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert len(block.insts) == 5
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 2  # 20 // 8
    assert all(isinstance(item, InstNode) for item in block.insts[1:])


def test_unroll_partial_unroll_no_remainder():
    """n=16, k=8 → IRLoop(n=2) wrapped in BlockNode, no remainder."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=16,
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert len(block.insts) == 1
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 2


def test_default_pipeline_can_disable_all_optimization_passes():
    """Disabling all pass flags should keep the IR layout unchanged."""
    Label.reset()
    root = RootNode(
        insts=[
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#1"})),
            InstNode(RegWriteInst(dst="s1", src="imm", extra_args={"LIT": "#2"})),
            InstNode(LabelInst(name=Label.make_new("dead"))),
            InstNode(TimeInst(c_op="inc_ref", lit="#0")),
            InstNode(TimeInst(c_op="inc_ref", lit="#1")),
            InstNode(TimeInst(c_op="inc_ref", lit="#2")),
        ]
    )

    pipeline = make_default_pipeline(pmem_capacity=8192)
    pipeline.config.disable_all_opt = True

    out, _ctx = pipeline(root)

    assert len(out.insts) == 6
    assert isinstance(cast(InstNode, out.insts[0]).inst, RegWriteInst)
    assert isinstance(cast(InstNode, out.insts[1]).inst, RegWriteInst)
    assert isinstance(cast(InstNode, out.insts[2]).inst, LabelInst)
    assert isinstance(cast(InstNode, out.insts[3]).inst, TimeInst)
    assert cast(TimeInst, cast(InstNode, out.insts[3]).inst).lit == "#0"
    assert isinstance(cast(InstNode, out.insts[4]).inst, TimeInst)
    assert cast(TimeInst, cast(InstNode, out.insts[4]).inst).lit == "#1"
    assert isinstance(cast(InstNode, out.insts[5]).inst, TimeInst)
    assert cast(TimeInst, cast(InstNode, out.insts[5]).inst).lit == "#2"


def test_unroll_no_scheduled_ticks_uses_zero_delay_budget():
    """Body has no inc_ref delay → scheduled_ticks=0, so k uses slack<=0 branch."""
    Label.reset()
    heavy_insts: list[IRNode] = [
        InstNode(PortWriteInst(dst="0", time="t0")) for _ in range(5)
    ]
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=100,
                body=BlockNode(insts=heavy_insts),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], BlockNode)
    block = cast(BlockNode, out.insts[0])
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 12  # 100 // max_unroll_factor(8)
    assert len(block.insts) == 1 + 4 * 5  # remainder=4, body_size=5


def test_unroll_dynamic_delay_only_body_uses_zero_delay_budget():
    """Body with only dynamic inc_ref → scheduled=0, so k uses slack<=0 branch."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(
                    insts=[InstNode(TimeInst(c_op="inc_ref", r1="r_delay"))]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], BlockNode)
    block = cast(BlockNode, out.insts[0])
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 2  # 20 // 8
    assert len(block.insts) == 1 + 4  # remainder=4


def test_unroll_mixed_literal_and_dynamic_delay_uses_literal_budget():
    """Mixed literal+dynamic inc_ref still treats the dynamic term as 0.
    Under the current large default jump cost, k hits the cap so n=4 fully
    expands."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=4,
                body=BlockNode(
                    insts=[
                        InstNode(TimeInst(c_op="inc_ref", lit="#5")),
                        InstNode(TimeInst(c_op="inc_ref", r1="r_dyn")),
                    ]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert _flat_inst_count(out) == 8  # 4 copies * 2 instructions


def test_unroll_exact_register_hint_fully_expands():
    """range_hint=(3,3) → treated as constant 3, fully expand into BlockNode."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                range_hint=(3, 3),
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert _flat_inst_count(out) == 3


def test_unroll_non_exact_register_hint_emits_jump_table():
    """range_hint=(2,5): non-exact register loop now goes through Phase 8D jump table.

    body=inc_ref #1 → scheduled=1, body_cost=1, slack=0 → k_timing=cap=8.
    body_size=1; floor_pow2(8) = 8 → IRJumpTableLoop(k=8).
    """
    from zcu_tools.program.v2.ir.passes.loop_dispatch import IRJumpTableLoop

    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                range_hint=(2, 5),
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRJumpTableLoop)
    jt = cast(IRJumpTableLoop, out.insts[0])
    assert jt.k == 8
    assert jt.n_reg == "r_count"
    assert jt.body_words == 1


def test_unroll_no_hint_register_loop_emits_jump_table():
    """Register-driven loop with no hint also goes to jump-table dispatch."""
    from zcu_tools.program.v2.ir.passes.loop_dispatch import IRJumpTableLoop

    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n="r_count",
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRJumpTableLoop)


def test_unroll_counter_sensitive_loop_still_uses_unroll_k_rules():
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=3,
                body=BlockNode(
                    insts=[
                        InstNode(TestInst(op="s0 - #1")),
                        InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                    ]
                ),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], BlockNode)
    assert _flat_inst_count(out) == 6


def test_unroll_partial_unroll_clones_labels_safely():
    Label.reset()
    inner = Label.make_new("inner")
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=20,
                body=BlockNode(
                    insts=[
                        InstNode(LabelInst(name=inner)),
                        InstNode(JumpInst(label=inner)),
                        InstNode(TimeInst(c_op="inc_ref", lit="#1")),
                    ]
                ),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=8)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    from zcu_tools.program.v2.ir.instructions import Instruction

    emit: list[Instruction] = []
    out.emit(emit)

    defined = {str(inst.name) for inst in emit if isinstance(inst, LabelInst)}
    targets = {
        str(inst.label) for inst in emit if isinstance(inst, JumpInst) and inst.label
    }

    internal_targets = {t for t in targets if not t.startswith("loop")}
    assert internal_targets <= defined


def test_unroll_budget_caps_k_below_timing():
    """body_size=1, pmem_budget=3 → k_budget=3. k_timing would be 8 (cap).
    k=min(8,3)=3. n=9 → iters=3, remainder=0 → IRLoop(n=3)."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=9,
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(pmem_budget=3)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 3


def test_unroll_max_factor_caps_k():
    """max_unroll_factor=2 caps k regardless of slack/budget. n=10 → iters=5."""
    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=10,
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(max_unroll_factor=2)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 5


def test_unroll_post_order_recurses_into_inner_loop_first():
    """Outer loop wraps an inner loop with n=2; both are constant. Inner unrolls
    first (post-order), then outer is evaluated against the rewritten body."""
    Label.reset()
    inner = IRLoop(
        name="inner",
        counter_reg="s1",
        n=2,
        body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
    )
    outer = IRLoop(
        name="outer",
        counter_reg="s0",
        n=2,
        body=BlockNode(insts=[inner]),
    )
    root = RootNode(insts=[outer])

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Both loops should be fully expanded → 4 inc_ref TimeInst.
    assert _flat_inst_count(out) == 4


def test_unroll_cpmg_style_body_triggers():
    """CPMG-style: 3 PortWrite + inc_ref #14.
    With the current large default jump cost, k reaches the cap 8.
    n=50 > k=8 → iters=6, remainder=2.
    body_size = 3 (PortWrite) + 1 (TimeInst) = 4.
    """
    Label.reset()
    body_insts: list[IRNode] = [
        InstNode(PortWriteInst(dst="0", time="t0")),
        InstNode(PortWriteInst(dst="1", time="t0")),
        InstNode(PortWriteInst(dst="2", time="t0")),
        InstNode(TimeInst(c_op="inc_ref", lit="#14")),
    ]
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="s0",
                n=50,
                body=BlockNode(insts=body_insts),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    block = out.insts[0]
    assert isinstance(block, BlockNode)
    assert isinstance(block.insts[0], IRLoop)
    assert cast(IRLoop, block.insts[0]).n == 6  # 50 // 8 = 6
    # Remainder: 50 % 8 = 2 → 2 * body_size(4) = 8 InstNodes after the loop.
    assert len(block.insts) == 1 + 2 * 4


# ---------------------------------------------------------------------------
# Phase 8E integration tests
# ---------------------------------------------------------------------------


def test_unroll_register_driven_k_forced_to_power_of_two():
    """With the current large default jump cost, k_raw reaches the cap 8 and
    remains 8 after floor_pow2()."""
    from zcu_tools.program.v2.ir.passes.loop_dispatch import IRJumpTableLoop

    Label.reset()
    body_insts: list[IRNode] = [
        InstNode(PortWriteInst(dst="0", time="t0")),
        InstNode(PortWriteInst(dst="1", time="t0")),
        InstNode(PortWriteInst(dst="2", time="t0")),
        InstNode(TimeInst(c_op="inc_ref", lit="#14")),
    ]
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=body_insts),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRJumpTableLoop)
    jt = cast(IRJumpTableLoop, out.insts[0])
    assert jt.k == 8


def test_unroll_register_driven_jump_table_structure():
    """Verify the IRJumpTableLoop has the expected k entries, k bodies, and
    the back-edge dispatch when emitted."""
    from zcu_tools.program.v2.ir.passes.loop_dispatch import IRJumpTableLoop

    Label.reset()
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
            )
        ]
    )

    config = PipeLineConfig(max_unroll_factor=2)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert isinstance(out.insts[0], IRJumpTableLoop)
    jt = cast(IRJumpTableLoop, out.insts[0])
    assert jt.k == 2
    assert len(jt.entry_labels) == 2
    assert len(jt.bodies) == 2
    assert jt.body_words == 1

    from zcu_tools.program.v2.ir.instructions import Instruction as _Inst

    emit: list[_Inst] = []
    out.emit(emit)
    # n==0 guard: TEST n - #0
    test_insts = [inst for inst in emit if isinstance(inst, TestInst)]
    assert any(t.op == "r_n - #0" for t in test_insts)
    # Back-edge dispatch test against literal #2 (k):
    assert any(t.op == "r_i - #2" for t in test_insts)


def test_unroll_register_driven_body_with_no_words_falls_back():
    """A body that flattens to body_words == 0 must fall back to no-unroll.

    Bare LabelInst contributes 0 to flat size and 0 to body_cost.
    """
    Label.reset()
    inner = Label.make_new("inner")
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=[InstNode(LabelInst(name=inner))]),
            )
        ]
    )

    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=PipeLineConfig()))

    # Falls back: original IRLoop unchanged.
    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_register_driven_dispatch_too_long_falls_back():
    """If body_words requires a shift-add sequence longer than max_dispatch_words,
    register-driven unroll must fall back to no-unroll."""
    # body_words = 0xFF requires 8 adds + 7 shifts = 15 instructions.
    # Use 0xFE NOPs plus one TIME word => 0xFF total body words.
    Label.reset()
    body_insts: list[IRNode] = [
        InstNode(NopInst()) for _ in range(0xFE)
    ]
    body_insts.append(InstNode(TimeInst(c_op="inc_ref", lit="#1000")))
    root = RootNode(
        insts=[
            IRLoop(
                name="loop",
                counter_reg="r_i",
                n="r_n",
                body=BlockNode(insts=body_insts),
            )
        ]
    )

    config = PipeLineConfig(max_dispatch_words=4)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    assert len(out.insts) == 1
    assert isinstance(out.insts[0], IRLoop)


def test_unroll_nested_register_driven_inner_unrolls_first():
    """Post-order recursion: the inner register-driven loop is rewritten to
    an IRJumpTableLoop before the outer loop is evaluated.
    """
    from zcu_tools.program.v2.ir.passes.loop_dispatch import IRJumpTableLoop

    Label.reset()
    inner = IRLoop(
        name="inner",
        counter_reg="r_j",
        n="r_m",
        body=BlockNode(insts=[InstNode(TimeInst(c_op="inc_ref", lit="#1"))]),
    )
    outer = IRLoop(
        name="outer",
        counter_reg="r_i",
        n="r_n",
        body=BlockNode(insts=[inner]),
    )
    root = RootNode(insts=[outer])

    config = PipeLineConfig(max_unroll_factor=2)
    out = UnrollSmallLoopPass().process(root, PipeLineContext(config=config))

    # With the current large default jump cost, the outer loop also gets
    # rewritten. The key property remains post-order recursion: the inner
    # loop rewrite must have happened before the outer body was measured.
    assert len(out.insts) == 1
    outer_out = out.insts[0]
    assert isinstance(outer_out, IRJumpTableLoop)


def test_default_pipeline_orders_new_passes_first():
    pipeline = make_default_pipeline(pmem_capacity=8192)

    assert [type(pass_).__name__ for pass_ in pipeline.passes] == [
        "UnrollSmallLoopPass",
        "DeadWriteEliminationPass",
        "DeadLabelEliminationPass",
        "ZeroDelayDCEPass",
        "TimedInstructionMergePass",
    ]
